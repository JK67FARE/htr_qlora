"""
Fine-tune Qwen VLMs for HTR with LoRA / QLoRA.

Expected data directory layout:
    <data_dir>/
        train.json
        validation.json # optional — falls back to the test split
        test.json
        train/          # images referenced by train.json
        validation/
        test/           # images referenced by test.json

Each JSON:  { "image_dir": "train", "num_samples": N,
              "samples": [{"image_file": "...", "transcription": "..."}, ...] }

Usage:
    python src/train_qwen_lora.py \
        --model-id Qwen/Qwen3.5-9B \
        --data-dir ./my_htr_data \
        --epochs 3 --lr 2e-4

    python src/train_qwen_lora.py \
        --model-id Qwen/Qwen3.5-9B \
        --data-dir ./my_htr_data \
        --use-qlora --epochs 3

Output dir & predictions JSON are auto-named
`<data_dir>_<model_id>_<lora|qlora>[_n<limit>]_s<seed>` — the training-sample
cap only appears when --limit is given. See --help for all options.

Attention runs on SDPA.

Checkpoint selection: after every epoch the *validation* split is transcribed
and scored with the CER of evaluate_htr.py (`base` normalisation, corpus level),
so the number that picks the checkpoint is the number the eval table reports.
Only epochs that improve it are checkpointed; the best adapter is restored at
the end and saved to --output-dir. The test split is touched exactly once, after
training, for the predictions JSON. Generating the whole validation split every
epoch is the expensive part of a run — budget --epochs accordingly.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# ── Pin to single GPU BEFORE torch initializes CUDA ───────
# Prevents HF Trainer from wrapping model in DataParallel.
# Override for multi-GPU: pass --device-map auto
if "--device-map" not in sys.argv:
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = _cvd.split(",")[0]

import editdistance  # noqa: E402  -- must follow the CUDA_VISIBLE_DEVICES pin above
import torch  # noqa: E402
from peft import LoraConfig, PeftModel  # noqa: E402
from qwen_vl_utils import process_vision_info as qwen_process_vision_info  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer  # noqa: E402

# Score with the same normalisation as the reporting script next door, so the
# CER here and the CER in the eval table are the same quantity.
from evaluate_htr import normalise_base, strip_preamble  # noqa: E402
from gpu_stats import (  # noqa: E402
    new_gpu_info,
    pass_entry,
    print_model_vram,
    reset_peak_vram,
    vram_snapshot,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI assistant specialized in transcribing handwritten text from images. Please follow these
guidelines:\

(1) Examine the image carefully and identify all handwritten text plus additional printed text that belongs to the handwritten content.\
(2) Transcribe ONLY the handwritten text and corresponding printed text. Ignore any additional printed or machine-generated text
in the image.\
(3) Maintain the original structure of the text, including line breaks and
paragraphs.\
(4) Do not attempt to correct spelling or grammar in the text. Transcribe it
exactly as written.\
(5) Do not describe the image or its contents.\
(6) Do not introduce or contextualize the transcription.\

Remember, your goal is to provide an accurate transcription of ONLY the handwritten and corresponding printed portions of the
text, preserving its original form as much as possible.
"""

USER_PROMPT = """\
Please transcribe the handwritten text in this image as accurately as possible, respecting line breaks.\
Every response should start with 'Transcription:', followed only by the transcription
"""


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen VLMs for HTR with LoRA / QLoRA"
    )

    # Model
    p.add_argument("--model-id", type=str, required=True,
                   help="HF model id, e.g. Qwen/Qwen3.5-9B")
    p.add_argument("--processor-id", type=str, default=None,
                   help="HF processor id. Defaults to <model_id>.")

    # Data — single directory
    p.add_argument("--data-dir", type=str, required=True,
                   help="Root directory containing train.json, test.json, optionally "
                        "validation.json, and the image folders.")
    p.add_argument("--max-train-samples", "--limit", dest="max_train_samples",
                   type=int, default=None,
                   help="Cap the number of training samples (shuffled with --seed "
                        "before truncating). Default: use all.")

    # LoRA / QLoRA
    p.add_argument("--use-qlora", action="store_true",
                   help="Enable 4-bit QLoRA. Without this flag, standard LoRA (bf16) is used.")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    p.add_argument("--lora-target-modules", type=str, default="all-linear",
                   help="Target modules for LoRA (comma-separated or 'all-linear').")

    # Training hyper-params
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: auto-named from dataset dir, "
                        "model id, method, sample cap and seed).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", type=str, default="cosine",
                   choices=["constant", "cosine", "linear"])
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--optim", type=str, default="adamw_torch_fused")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--report-to", type=str, default="tensorboard",
                   choices=["tensorboard", "wandb", "none"])
    p.add_argument("--max-length", type=int, default=512,
                   help="Maximum sequence length (tokens).")

    # Generation / eval
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max new tokens during eval generation.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for eval (0 = greedy).")
    p.add_argument("--prediction-key", type=str, default=None,
                   help="Key name for predictions in the output JSON "
                        "(default: 'qlora_zs' or 'lora_zs' by method).")

    # Prompt
    p.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT,
                   help="System prompt for the model.")
    p.add_argument("--user-prompt", type=str, default=USER_PROMPT,
                   help="User prompt prepended to each image.")

    # Post-training
    p.add_argument("--merge-and-export", action="store_true",
                   help="After training, merge adapter into base model and save.")

    # Misc
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device-map", type=str, default=None,
                   help="Device map. Default: single GPU. Use 'auto' for multi-GPU.")

    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = run_slug(args)

    # The prediction key names the method, so LoRA and QLoRA predictions stay
    # distinguishable after they are merged into one eval table.
    if args.prediction_key is None:
        args.prediction_key = f"{method_slug(args)}_zs"

    return args


def limit_slug(args) -> str:
    """'_n<limit>' when --max-train-samples/--limit is set, else ''."""
    return f"_n{args.max_train_samples}" if args.max_train_samples is not None else ""


def method_slug(args) -> str:
    """'qlora' with --use-qlora, else 'lora'."""
    return "qlora" if args.use_qlora else "lora"


def run_slug(args) -> str:
    """
    '<dataset-dir>_<model-id>_<method>[_n<limit>]_s<seed>' — everything that
    distinguishes one run from another, so two runs never write into the same
    directory or over each other's predictions JSON. The sample cap is only
    part of the name when --max-train-samples/--limit is actually given.
    """
    dataset_slug = Path(args.data_dir).resolve().name
    model_slug = args.model_id.rstrip("/").split("/")[-1]
    return f"{dataset_slug}_{model_slug}_{method_slug(args)}{limit_slug(args)}_s{args.seed}"


# ---------------------------------------------------------------------------
# 2. Dataset helpers
# ---------------------------------------------------------------------------

def load_manifest(json_path: str | Path) -> tuple[list[dict], str]:
    """Load a JSON manifest → (list of sample dicts, image dir as a string)."""
    json_path = Path(json_path)
    if not json_path.exists():
        sys.exit(f"ERROR: {json_path} not found")
    with open(json_path) as f:
        manifest = json.load(f)

    # split_data.py writes `image_dir` relative to the CWD it ran in
    # ("<dataset>/train"), so take its last component next to the manifest.
    image_dir = json_path.parent / Path(manifest["image_dir"]).name
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


def limit_samples(samples: list[dict], limit: int | None, seed: int) -> list[dict]:
    """
    Shuffle with *seed* and keep the first *limit* samples.

    Same seed + same limit + same manifest → same subset here and in
    train_hf.py, so the two stay comparable on identical data.
    """
    if limit is None or limit >= len(samples):
        return samples
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    print(f"Limited training set to {limit} samples (seed {seed})")
    return samples[:limit]


def samples_to_dataset(samples: list[dict]):
    """Convert list of sample dicts → HF Dataset."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        "image_path":    [s["image_path"]    for s in samples],
        "image_file":    [s["image_file"]    for s in samples],
        "transcription": [s["transcription"] for s in samples],
    })


# ---------------------------------------------------------------------------
# 2b. Message building & image extraction
# ---------------------------------------------------------------------------

def build_messages(image_path: str, transcription: str | None,
                   system_prompt: str, user_prompt: str,
                   include_assistant: bool = True) -> list[dict]:
    """Build multi-modal chat messages with a file:// image reference
    (qwen_vl_utils loads the image when the processor sees it)."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    if include_assistant and transcription is not None:
        msgs.append({
            "role": "assistant",
            "content": [{"type": "text", "text": transcription}],
        })
    return msgs


def extract_images(messages: list[dict]):
    """Extract images/videos from messages via qwen_vl_utils.
    Returns (images_list, videos_list); either may be None."""
    images, videos = qwen_process_vision_info(messages)
    return images or None, videos or None


def call_processor(processor, text, images, videos=None, **kwargs):
    """Call the processor, passing videos only when present."""
    proc_kwargs = dict(text=text, images=images, **kwargs)
    if videos is not None:
        proc_kwargs["videos"] = videos
    return processor(**proc_kwargs)


def apply_chat_template(processor, messages, **kwargs):
    """Apply chat template; disable Qwen3 thinking mode for HTR."""
    kwargs.setdefault("enable_thinking", False)
    return processor.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# 2c. Universal label masking
# ---------------------------------------------------------------------------

def find_assistant_content_start(input_ids: list[int], tokenizer) -> int | None:
    """Find the token index where the assistant's actual content begins.

    Looks for Qwen's <|im_start|> header followed by "assistant" tokens.
    Falls back to searching for "assistant" text as last resort.

    Returns the index of the first content token after the header, or None.
    """
    n = len(input_ids)

    patterns = [
        # (header_token_str, role_text)
        ("<|im_start|>", "assistant"),   # Qwen
    ]

    for header_str, role_text in patterns:
        header_id = tokenizer.convert_tokens_to_ids(header_str)
        if header_id == tokenizer.unk_token_id:
            continue  # this token doesn't exist in this tokenizer

        role_ids = tokenizer.encode(role_text, add_special_tokens=False)
        role_len = len(role_ids)

        # Search backwards for the LAST occurrence (handles multi-turn)
        for pos in range(n - 1, -1, -1):
            if input_ids[pos] == header_id:
                # Content starts after header + role tokens + newline
                content_start = pos + 1 + role_len
                # Skip a trailing newline token if present
                if content_start < n:
                    nl_text = tokenizer.decode([input_ids[content_start]])
                    if nl_text.strip() == "":
                        content_start += 1
                return min(content_start, n)

    # --- Fallback: search for "assistant" text tokens ---
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    for pos in range(n - len(assistant_ids), -1, -1):
        if input_ids[pos:pos + len(assistant_ids)] == assistant_ids:
            return pos + len(assistant_ids)

    return None


# ---------------------------------------------------------------------------
# 3. Evaluation: CER & WER → predictions JSON
# ---------------------------------------------------------------------------

# Corpus-level CER on `base`-normalised text, exactly as evaluate_htr.py
# computes it: the leading "Transcription:" label the prompt asks for is
# forgiven, everything else the model emits counts as an error.
#
# This exists to pick a checkpoint, not to report a score — nothing here is
# written to the predictions JSON. Run evaluate_htr.py on that file for the
# numbers that go in a table (all normalisation variants, WER, accuracy).


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    """Σ character edit distance / Σ reference characters."""
    errors = total = 0
    for pred, ref in zip(preds, refs):
        p, r = normalise_base(strip_preamble(pred)), normalise_base(ref)
        errors += editdistance.eval(p, r)
        total += len(r)
    return errors / total if total else 0.0


def predictions_filename(args) -> str:
    """
    Return '<dataset-dir>_<model-id>_<method>[_n<limit>]_s<seed>_predictions.json'
    — the same slug as the output dir, so the file stays identifiable once it is
    copied out of it.
    """
    return f"{run_slug(args)}_predictions.json"


@torch.no_grad()
def transcribe_samples(model, processor, samples: list[dict], args,
                       desc: str) -> list[str]:
    """Transcribe every sample, in manifest order."""
    was_training = model.training
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        # Training turns the KV cache off for gradient checkpointing; generation
        # without it re-encodes the prompt for every token.
        use_cache=True,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95

    preds: list[str] = []
    for sample in tqdm(samples, desc=desc, unit="img"):
        messages = build_messages(
            sample["image_path"], None,
            args.system_prompt, args.user_prompt,
            include_assistant=False,
        )
        text = apply_chat_template(
            processor, messages,
            tokenize=False, add_generation_prompt=True,
        )
        images, videos = extract_images(messages)
        inputs = call_processor(
            processor, text=[text], images=images, videos=videos,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        preds.append(processor.tokenizer.decode(
            generated_ids_trimmed, skip_special_tokens=True
        ).strip())

    model.train(was_training)
    return preds


class CERSFTTrainer(SFTTrainer):
    """
    SFTTrainer that reports `eval_cer` next to `eval_loss`.

    The metric needs free-running generation, which the Trainer eval loop does
    not do, so the validation split is decoded here with the same helper the
    final test evaluation uses. Returning the value from evaluate() is what lets
    Trainer's `metric_for_best_model="eval_cer"` keep the best epoch.
    """

    def __init__(self, *trainer_args, val_samples: list[dict], cli_args, **kwargs):
        super().__init__(*trainer_args, **kwargs)
        self.val_samples = val_samples
        self.cli_args = cli_args
        self.val_refs = [s["transcription"] for s in val_samples]

    def evaluate(self, eval_dataset=None, ignore_keys=None,
                 metric_key_prefix="eval") -> dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        preds = transcribe_samples(self.model, self.processing_class,
                                   self.val_samples, self.cli_args, desc="val CER")
        key = f"{metric_key_prefix}_cer"
        metrics[key] = round(corpus_cer(preds, self.val_refs), 6)
        self.log({key: metrics[key]})
        return metrics


def run_test_eval(model, processor, test_samples: list[dict], args,
                  image_dir_str: str, training_info: dict, gpu_info: dict) -> dict:
    """Generate on the test set and write the predictions JSON.

    Scoring is evaluate_htr.py's job — this writes predictions and run metadata
    only, so there is exactly one place where a reported CER/WER comes from."""
    n = len(test_samples)

    print(f"\n{'='*60}")
    print(f"Evaluating on {n} test samples ...")
    print(f"{'='*60}")

    # Peaks are measured for the inference pass alone: training peaks much
    # higher (optimiser state + activations), and the vram_gib column in the
    # eval table is about what it costs to *run* the adapter.
    reset_peak_vram()
    t0 = time.time()
    preds = transcribe_samples(model, processor, test_samples, args, desc="test")
    elapsed = time.time() - t0
    gpu_info["peak_per_pass"][args.prediction_key] = pass_entry(vram_snapshot())

    refs = [s["transcription"].strip() for s in test_samples]
    predictions = {s["image_file"]: {args.prediction_key: pred}
                   for s, pred in zip(test_samples, preds)}

    for i, (ref, pred) in enumerate(zip(refs, preds)):
        if i < 5 or (i + 1) % 50 == 0:
            print(f"\n--- Sample {i+1}/{n} ---")
            print(f"  REF : {ref[:120]}{'…' if len(ref) > 120 else ''}")
            print(f"  PRED: {pred[:120]}{'…' if len(pred) > 120 else ''}")
            print(f"  CER : {corpus_cer([pred], [ref]):.4f}")

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
                args.prediction_key: {
                    "duration_s": round(elapsed, 1),
                    "images": n,
                    "s_per_image": round(elapsed / max(n, 1), 2),
                    "entries_per_second": round(max(n, 1) / max(elapsed, 0.01), 3),
                },
                "training_s": train_duration_s,
                "inference_s": round(elapsed, 1),
                "total_s": round(elapsed + train_duration_s, 1),
            },
        },
    }

    json_path = Path(args.output_dir) / predictions_filename(args)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  {n} predictions  –  {elapsed:.1f}s inference "
          f"({elapsed/max(n,1):.2f}s / image)")
    print(f"  {train_duration_s:.1f}s training")
    print(f"  Predictions written to: {json_path}")
    print(f"  Score them:  python evaluate_htr.py --predictions {json_path} "
          f"--annotations <annotations.json>")
    print(f"{'='*60}\n")
    return output


# ---------------------------------------------------------------------------
# 4. Main — training entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Using GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

    data_dir = Path(args.data_dir)

    # ── Processor ──────────────────────────────────────────────
    processor_id = args.processor_id or args.model_id
    processor = AutoProcessor.from_pretrained(processor_id)

    # ── Model kwargs ───────────────────────────────────────────
    device_map = args.device_map or {"": 0}
    model_kwargs: dict = dict(
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=device_map,
        # SDPA ships with torch — no flash-attn build to keep in sync with the
        # CUDA/torch version, and it is what transformers would pick anyway.
        attn_implementation="sdpa",
    )

    if args.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["dtype"],
            bnb_4bit_quant_storage=model_kwargs["dtype"],
        )

    print(f"Loading model: {args.model_id}  ({'QLoRA' if args.use_qlora else 'LoRA'})")
    reset_peak_vram()
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    gpu_info = new_gpu_info(model)
    print_model_vram(gpu_info["model"])

    # ── LoRA config ────────────────────────────────────────────
    target_modules = (
        args.lora_target_modules
        if args.lora_target_modules == "all-linear"
        else [m.strip() for m in args.lora_target_modules.split(",")]
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # ── Dataset ────────────────────────────────────────────────
    print("Loading training data …")
    train_raw, _ = load_manifest(data_dir / "train.json")
    train_raw = limit_samples(train_raw, args.max_train_samples, args.seed)
    train_dataset = samples_to_dataset(train_raw)

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
    val_dataset = samples_to_dataset(val_raw)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}  "
          f"|  Test samples: {len(test_raw)}")

    # ── Collator (loads images lazily) ─────────────────────────
    def collate_fn(examples):
        texts, all_images, all_videos = [], [], []
        for ex in examples:
            messages = build_messages(
                ex["image_path"], ex["transcription"],
                args.system_prompt, args.user_prompt,
                include_assistant=True,
            )
            text = apply_chat_template(
                processor, messages,
                add_generation_prompt=False, tokenize=False,
            )
            images, videos = extract_images(messages)
            texts.append(text.strip())
            all_images.append(images)
            if videos is not None:
                all_videos.append(videos)

        # Flatten image lists for the processor
        flat_images = []
        for img_list in all_images:
            if img_list:
                flat_images.extend(img_list) if isinstance(img_list, list) else flat_images.append(img_list)

        proc_kwargs = dict(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
        )
        if all_videos:
            flat_videos = []
            for v in all_videos:
                flat_videos.extend(v) if isinstance(v, list) else flat_videos.append(v)
            proc_kwargs["videos"] = flat_videos

        batch = processor(**proc_kwargs)

        # ── Build labels: mask pad + everything before assistant turn ──
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for i in range(labels.size(0)):
            id_list = batch["input_ids"][i].tolist()
            start = find_assistant_content_start(id_list, processor.tokenizer)
            if start is not None:
                labels[i, :start] = -100

        batch["labels"] = labels
        return batch

    # ── Training config ────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        seed=args.seed,
        max_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        # Score the validation split once per epoch and checkpoint only when its
        # CER improves, so the one checkpoint left on disk is the best epoch and
        # its adapter is what gets restored and saved.
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
    )

    trainer = CERSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
        val_samples=val_raw,
        cli_args=args,
    )

    # ── Train ──────────────────────────────────────────────────
    print(f"\nStarting training → {args.output_dir}")
    reset_peak_vram()
    train_t0 = time.time()
    trainer.train()
    # Recorded separately from the inference peak: this is the number that says
    # whether the run would have fitted on a smaller card.
    gpu_info["peak_per_pass"]["training"] = pass_entry(vram_snapshot())

    # trainer.train() has already put the best epoch's adapter back into the
    # model (load_best_model_at_end), so this writes the best one, not the last.
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        print("WARNING: no checkpoint was written — saving the final adapter",
              file=sys.stderr)
    else:
        print(f"Best checkpoint: {Path(best_ckpt).name}  "
              f"(val CER = {trainer.state.best_metric:.4f}) — restored")
    trainer.save_model()
    train_duration_s = time.time() - train_t0
    print(f"Training finished in {train_duration_s:.1f}s")

    # ── Clean up output dir: keep only adapter + processor ─────
    output_path = Path(args.output_dir)
    keep_files = {
        "adapter_model.safetensors", "adapter_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "preprocessor_config.json",
        "chat_template.json", "config.json",
        "merges.txt", "vocab.json",  # Qwen BPE
    }
    for item in list(output_path.iterdir()):
        if item.name in keep_files or item.name.endswith("_predictions.json"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    print(f"Adapter saved to: {args.output_dir}")

    # ── Evaluate CER / WER on the test split → predictions JSON ─
    run_test_eval(model, processor, test_raw, args,
                  image_dir_str=test_image_dir,
                  training_info={
                      "duration_s": round(train_duration_s, 1),
                      # Actual count, not the cap: --limit above the split size
                      # silently trains on fewer samples than the name suggests.
                      "train_samples": len(train_dataset),
                      "max_train_samples": args.max_train_samples,
                      "seed": args.seed,
                      "val_split": val_split,
                      "best_val_cer": trainer.state.best_metric,
                      "best_checkpoint": Path(best_ckpt).name if best_ckpt else None,
                  },
                  gpu_info=gpu_info)

    # ── Merge & export ─────────────────────────────────────────
    if args.merge_and_export:
        merged_dir = f"{args.output_dir}-merged"
        print(f"\nMerging adapter into base model → {merged_dir}")

        del model
        del trainer
        torch.cuda.empty_cache()

        base_model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16 if args.bf16 else torch.float16,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")

        merged_processor = AutoProcessor.from_pretrained(args.output_dir)
        merged_processor.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
