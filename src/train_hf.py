"""
Fine-tune TrOCR (microsoft/trocr-base-handwritten) for Handwritten Text
Recognition on the same data layout used by train_qwen_lora.py.

Expected data directory layout:
    <data_dir>/
        train.json
        validation.json     # optional — falls back to the test split
        test.json
        train/              # images referenced by train.json
        validation/
        test/

Each JSON:  { "image_dir": "train", "num_samples": N,
              "samples": [{"image_file": "...", "transcription": "..."}, ...] }

Usage:
    python src/train_hf.py \
        --data-dir metadata_crops_2000_tvt \
        --epochs 10 --lr 1e-5

Checkpoint selection: after every epoch the *validation* split is transcribed
and scored with the CER of evaluate_htr.py (`base` normalisation, corpus
level), so the number that picks the checkpoint is the number the eval table
reports. Only epochs that improve that CER are checkpointed, and the best one is
reloaded at the end and written to --output-dir. The test split is touched
exactly once, after training, for the predictions JSON.

Peak VRAM is recorded per pass under remarks.gpu.peak_per_pass — once for
training (the loop, per-epoch validation included) and once for inference — in
the shape evaluate_htr.py reads, so both are comparable with the prediction
runs.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Pin to single GPU BEFORE torch initializes CUDA ───────
# Prevents HF Trainer from wrapping the model in DataParallel.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _cvd.split(",")[0]

import editdistance  # noqa: E402  -- must follow the CUDA_VISIBLE_DEVICES pin above
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

# Reuse the scoring script's normalisation so the CER here and the CER in the
# eval table are the same quantity, and the shared VRAM accounting so the
# numbers mean the same thing for a fine-tune as for a prediction run.
from evaluate_htr import normalise_base, strip_preamble  # noqa: E402
from gpu_stats import (  # noqa: E402
    gpu_devices_info,
    pass_entry,
    reset_peak_vram,
    total_peak_reserved_gib,
    vram_snapshot,
)


# ---------------------------------------------------------------------------
# 0. Fixed policy — deliberately not on the CLI
# ---------------------------------------------------------------------------
# These stay identical across runs so two runs differ only in data, epochs and
# learning rate. Change them here (and rerun everything) if the policy changes.

MAX_TARGET_LENGTH = 64          # label cap in tokens; also the generation cap
NUM_BEAMS = 1                   # greedy decoding
PREDICTION_KEY = "trocr_ft"     # pass label inside the predictions JSON
LR_SCHEDULER = "linear"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_torch_fused"
LOGGING_STEPS = 10
REPORT_TO = "tensorboard"
DATALOADER_WORKERS = 4


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full fine-tune of TrOCR for HTR"
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="Root directory containing train.json, test.json, "
                        "optionally validation.json, and the image folders.")
    p.add_argument("--model-id", type=str, default="microsoft/trocr-base-handwritten",
                   help="HF model id of a VisionEncoderDecoder OCR model.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-device batch size, used for training and for generation.")
    p.add_argument("--max-train-samples", "--limit", dest="max_train_samples",
                   type=int, default=None,
                   help="Cap the number of training samples (shuffled with --seed "
                        "before truncating). Default: use all.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: auto-named from dataset, "
                        "model id, sample cap, epochs and seed).")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # ── Derive output_dir from everything that distinguishes a run ─
    # Two runs that differ in data, sample cap, epochs or seed must not write
    # into the same directory, so all four go into the name.
    if args.output_dir is None:
        dataset_slug = Path(args.data_dir).resolve().name
        model_slug = args.model_id.rstrip("/").split("/")[-1]
        limit_slug = (f"n{args.max_train_samples}"
                      if args.max_train_samples is not None else "nall")
        args.output_dir = "_".join([dataset_slug, model_slug, limit_slug,
                                    f"e{args.epochs}", f"s{args.seed}"])

    return args


# ---------------------------------------------------------------------------
# 2. Dataset helpers  (mirrors train_qwen_lora.py)
# ---------------------------------------------------------------------------

def resolve_image_dir(json_path: Path, manifest_image_dir: str) -> Path:
    """
    Locate the image folder a manifest points at.

    split_data.py writes `image_dir` relative to the CWD it happened to run
    in, so older manifests say "./train" and newer ones "<dataset>/train". Try
    the manifest-relative path, then the basename next to the manifest, then the
    string as given.
    """
    candidates = [json_path.parent / manifest_image_dir,
                  json_path.parent / Path(manifest_image_dir).name,
                  Path(manifest_image_dir)]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    sys.exit(f"ERROR: image dir '{manifest_image_dir}' of {json_path} not found "
             f"(tried: {', '.join(str(c) for c in candidates)})")


def load_manifest(json_path: str | Path) -> tuple[list[dict], str]:
    """Load a JSON manifest → (list of sample dicts, image dir as a string)."""
    json_path = Path(json_path)
    if not json_path.exists():
        sys.exit(f"ERROR: {json_path} not found")
    with open(json_path) as f:
        manifest = json.load(f)

    image_dir = resolve_image_dir(json_path, manifest["image_dir"])
    samples = []
    for s in manifest["samples"]:
        img_path = image_dir / s["image_file"]
        if not img_path.exists():
            print(f"WARNING: image not found, skipping: {img_path}", file=sys.stderr)
            continue
        samples.append({
            "image_path": str(img_path),
            "image_file": s["image_file"],
            "transcription": s["transcription"],
        })
    if not samples:
        sys.exit(f"ERROR: no usable samples in {json_path}")
    return samples, str(image_dir)


class HTRDataset(Dataset):
    """Images → pixel_values, transcriptions → padded label ids (pad = -100)."""

    def __init__(self, samples: list[dict], processor: TrOCRProcessor,
                 eos_token_id: int | None = None):
        self.samples = samples
        self.processor = processor
        self.eos_token_id = (eos_token_id if eos_token_id is not None
                             else processor.tokenizer.sep_token_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        pixel_values = self.processor.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values[0]

        # TrOCR's decoder starts from </s> and then emits the text directly — it
        # never predicts a leading <s>. Adding the tokenizer's default BOS here
        # would shift every label by one position against that convention (it
        # roughly doubles the initial loss), so build the sequence by hand:
        #   labels = [text tokens ..., </s>], padded with -100.
        ids = self.processor.tokenizer(
            s["transcription"], add_special_tokens=False
        ).input_ids
        ids = ids[: MAX_TARGET_LENGTH - 1] + [self.eos_token_id]
        labels = ids + [-100] * (MAX_TARGET_LENGTH - len(ids))

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# ---------------------------------------------------------------------------
# 3. Metrics — corpus CER/WER on `base`-normalised text, as in evaluate_htr.py
# ---------------------------------------------------------------------------

def corpus_cer(preds: list[str], refs: list[str]) -> float:
    """Σ character edit distance / Σ reference characters."""
    errors = total = 0
    for pred, ref in zip(preds, refs):
        p, r = normalise_base(strip_preamble(pred)), normalise_base(ref)
        errors += editdistance.eval(p, r)
        total += len(r)
    return errors / total if total else 0.0


def corpus_wer(preds: list[str], refs: list[str]) -> float:
    """Σ word edit distance / Σ reference words."""
    errors = total = 0
    for pred, ref in zip(preds, refs):
        p = normalise_base(strip_preamble(pred)).split()
        r = normalise_base(ref).split()
        errors += editdistance.eval(p, r)
        total += len(r)
    return errors / total if total else 0.0


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def transcribe(model, processor, samples: list[dict], batch_size: int,
               desc: str) -> list[str]:
    """Greedy-decode every sample, in manifest order."""
    was_training = model.training
    model.eval()
    preds: list[str] = []
    for start in tqdm(range(0, len(samples), batch_size), desc=desc, unit="batch"):
        batch = samples[start:start + batch_size]
        images = [Image.open(s["image_path"]).convert("RGB") for s in batch]
        pixel_values = processor.image_processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(model.device, dtype=model.dtype)

        generated_ids = model.generate(
            pixel_values,
            max_length=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
        )
        preds += [t.strip() for t in processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)]
    model.train(was_training)
    return preds


# ---------------------------------------------------------------------------
# 5. Trainer — validation CER after every epoch
# ---------------------------------------------------------------------------

class CERSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that reports `eval_cer` alongside `eval_loss`.

    HF's own `predict_with_generate` path hands the batch's `labels` to
    generate(), which VisionEncoderDecoder does not tolerate, so the validation
    split is decoded here with the same helper the final test evaluation uses.
    Returning the value from evaluate() is what lets Trainer's
    `metric_for_best_model="eval_cer"` keep the best epoch.
    """

    def __init__(self, *args, val_samples: list[dict], val_batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_samples = val_samples
        self.val_batch_size = val_batch_size
        self.val_refs = [s["transcription"] for s in val_samples]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval",
                 **gen_kwargs) -> dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix,
                                   **gen_kwargs)
        preds = transcribe(self.model, self.processing_class, self.val_samples,
                           self.val_batch_size, desc="val CER")
        key = f"{metric_key_prefix}_cer"
        metrics[key] = round(corpus_cer(preds, self.val_refs), 6)
        self.log({key: metrics[key]})
        return metrics


# ---------------------------------------------------------------------------
# 6. Final evaluation: CER & WER on the test split → predictions JSON
# ---------------------------------------------------------------------------

def run_test_eval(model, processor, test_samples: list[dict], args,
                  image_dir_str: str, training_info: dict, gpu_info: dict) -> dict:
    """Generate on the test set, compute CER/WER, write predictions JSON."""
    n = len(test_samples)

    print(f"\n{'='*60}")
    print(f"Evaluating on {n} test samples ...")
    print(f"{'='*60}")

    # Peaks are measured per pass, so the inference figure is what generation
    # alone needed — not whatever training left as the high-water mark.
    reset_peak_vram()
    t0 = time.time()
    preds = transcribe(model, processor, test_samples, args.batch_size, desc="test")
    elapsed = time.time() - t0
    infer_vram = vram_snapshot()
    gpu_info["peak_per_pass"][PREDICTION_KEY] = pass_entry(infer_vram)

    refs = [s["transcription"].strip() for s in test_samples]
    predictions = {s["image_file"]: {PREDICTION_KEY: pred}
                   for s, pred in zip(test_samples, preds)}

    for idx, (ref, pred) in enumerate(zip(refs, preds)):
        if idx < 5 or (idx + 1) % 50 == 0:
            print(f"\n--- Sample {idx+1}/{n} ---")
            print(f"  REF : {ref[:120]}{'…' if len(ref) > 120 else ''}")
            print(f"  PRED: {pred[:120]}{'…' if len(pred) > 120 else ''}")
            print(f"  CER : {corpus_cer([pred], [ref]):.4f}")

    cer, wer = corpus_cer(preds, refs), corpus_wer(preds, refs)
    train_duration_s = training_info["duration_s"]

    output = {
        "image_dir": image_dir_str,
        "model": args.model_id,
        "scale_factor": 1.0,
        "predictions": predictions,
        "remarks": {
            "parameters": vars(args),
            "training": training_info,
            "gpu": gpu_info,
            "runtime": {
                PREDICTION_KEY: {
                    "duration_s": round(elapsed, 1),
                    "images": n,
                    "s_per_image": round(elapsed / max(n, 1), 2),
                    "entries_per_second": round(max(n, 1) / max(elapsed, 0.01), 3),
                },
                "training_s": train_duration_s,
                "inference_s": round(elapsed, 1),
                "total_s": round(elapsed + train_duration_s, 1),
            },
            # Corpus-level, `base` normalisation — the same definition
            # evaluate_htr.py reports. Score properly with that script.
            "metrics": {
                "cer": round(cer, 6),
                "wer": round(wer, 6),
            },
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model_id.rstrip("/").split("/")[-1]
    json_path = out_dir / f"{model_slug}_ft_predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    train_peak = gpu_info["peak_per_pass"].get("training", {}).get(
        "peak_reserved_total_GiB")
    infer_peak = gpu_info["peak_per_pass"][PREDICTION_KEY]["peak_reserved_total_GiB"]

    print(f"\n{'='*60}")
    print(f"  RESULTS  –  CER: {cer:.4f}  |  WER: {wer:.4f}   (corpus, base)")
    print(f"  {elapsed:.1f}s inference  ({elapsed/max(n,1):.2f}s / image)")
    print(f"  {train_duration_s:.1f}s training")
    print(f"  peak VRAM  –  training: "
          f"{'n/a' if train_peak is None else f'{train_peak:.2f} GiB'}  |  "
          f"inference: {infer_peak:.2f} GiB   (reserved)")
    print(f"  Predictions written to: {json_path}")
    print(f"{'='*60}\n")
    return output


# ---------------------------------------------------------------------------
# 7. Model setup
# ---------------------------------------------------------------------------

def load_model(args, processor):
    """Load the VisionEncoderDecoder model and wire up the special tokens."""
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)

    tok = processor.tokenizer
    gen = model.generation_config

    # TrOCR (fairseq heritage) starts decoding from </s>, not <s>. Take the
    # checkpoint's own ids rather than guessing from the tokenizer, so the
    # fine-tune stays in the pretrained decoder's convention. The tokenizer is
    # only a fallback for checkpoints that ship an incomplete generation config.
    decoder_start_id = gen.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = tok.cls_token_id
    pad_id = gen.pad_token_id if gen.pad_token_id is not None else tok.pad_token_id
    eos_id = gen.eos_token_id if gen.eos_token_id is not None else tok.sep_token_id

    # VisionEncoderDecoderModel.forward() reads these off .config when it shifts
    # labels into decoder_input_ids, and they are absent by default.
    model.config.decoder_start_token_id = decoder_start_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    model.config.vocab_size = model.config.decoder.vocab_size

    gen.decoder_start_token_id = decoder_start_id
    gen.pad_token_id = pad_id
    gen.eos_token_id = eos_id
    gen.max_length = MAX_TARGET_LENGTH
    gen.num_beams = NUM_BEAMS
    gen.early_stopping = NUM_BEAMS > 1
    # Repeated tokens are legitimate in transcriptions — don't block them.
    gen.no_repeat_ngram_size = 0
    gen.length_penalty = 1.0

    return model


# ---------------------------------------------------------------------------
# 8. Main — training entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Using GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

    data_dir = Path(args.data_dir)

    # ── Processor & model ──────────────────────────────────────
    processor = TrOCRProcessor.from_pretrained(args.model_id)

    print(f"Loading model: {args.model_id}  (full fine-tune)")
    reset_peak_vram()
    model = load_model(args, processor)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    load_vram = vram_snapshot()
    gpu_info = {
        "devices": gpu_devices_info(),
        "model": {
            "after_load": load_vram,
            "load_peak_reserved_GiB": total_peak_reserved_gib(load_vram),
        },
        # Filled in below for "training" and in run_test_eval() for the
        # inference pass, so the two can be compared side by side.
        "peak_per_pass": {},
    }

    # ── Dataset ────────────────────────────────────────────────
    print("Loading training data …")
    train_raw, _ = load_manifest(data_dir / "train.json")

    if args.max_train_samples is not None and args.max_train_samples < len(train_raw):
        random.Random(args.seed).shuffle(train_raw)
        train_raw = train_raw[:args.max_train_samples]
        print(f"Limited training set to {len(train_raw)} samples (seed {args.seed})")

    print("Loading test data …")
    test_raw, test_image_dir = load_manifest(data_dir / "test.json")

    # Checkpoints are picked on the validation split. Datasets without one fall
    # back to the test split, which makes the reported test CER optimistic.
    val_json = data_dir / "validation.json"
    if val_json.exists():
        print("Loading validation data …")
        val_raw, _ = load_manifest(val_json)
        val_split = "validation"
    else:
        print(f"WARNING: no {val_json} — picking the best epoch on the TEST split",
              file=sys.stderr)
        val_raw = test_raw
        val_split = "test (no validation.json)"

    eos_id = model.config.eos_token_id
    train_dataset = HTRDataset(train_raw, processor, eos_id)
    val_dataset = HTRDataset(val_raw, processor, eos_id)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}  "
          f"|  Test samples: {len(test_raw)}")

    # ── Training config ────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIM,
        bf16=torch.cuda.is_available(),
        logging_steps=LOGGING_STEPS,
        report_to=REPORT_TO,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=DATALOADER_WORKERS,
        # Score the validation split once per epoch and checkpoint only when its
        # CER improves, so the single checkpoint left on disk (~1.3 GB) is the
        # best epoch. It is reloaded by hand below — see the comment there for
        # why load_best_model_at_end is not used.
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        predict_with_generate=False,  # CER comes from CERSeq2SeqTrainer.evaluate()
    )

    trainer = CERSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        processing_class=processor,
        val_samples=val_raw,
        val_batch_size=args.batch_size,
    )

    # ── Train ──────────────────────────────────────────────────
    print(f"\nStarting training → {args.output_dir}")
    reset_peak_vram()
    train_t0 = time.time()
    trainer.train()
    train_duration_s = time.time() - train_t0

    # The peak covers the whole loop, per-epoch validation generation included:
    # that is what the run had to be given to train.
    train_vram = vram_snapshot()
    gpu_info["peak_per_pass"]["training"] = pass_entry(train_vram)
    print(f"Training finished in {train_duration_s:.1f}s  "
          f"(peak VRAM {gpu_info['peak_per_pass']['training']['peak_reserved_total_GiB']:.2f} GiB "
          f"reserved)")

    # ── Restore the best epoch ─────────────────────────────────
    # Not via load_best_model_at_end: transformers writes VisionEncoderDecoder
    # checkpoints under the legacy ViT key names, and Trainer restores them with
    # a plain load_state_dict that does not translate those names back. It logs
    # the ~160 keys it dropped and carries on, leaving the encoder at the *last*
    # epoch's weights. from_pretrained does the translation, so reload with it.
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        print("WARNING: no checkpoint was written — keeping the final weights",
              file=sys.stderr)
    else:
        print(f"Best checkpoint: {best_ckpt}  "
              f"(val CER = {trainer.state.best_metric:.4f}) — reloading")
        model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(model.device)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    # ── Evaluate CER / WER on the test split → predictions JSON ─
    run_test_eval(model, processor, test_raw, args,
                  image_dir_str=test_image_dir,
                  gpu_info=gpu_info,
                  training_info={
                      "duration_s": round(train_duration_s, 1),
                      "val_split": val_split,
                      "best_val_cer": trainer.state.best_metric,
                      "best_checkpoint": trainer.state.best_model_checkpoint,
                  })

    print("\nDone.")


if __name__ == "__main__":
    main()
