"""
VLM HTR LoRA / QLoRA fine-tuning on HuggingFace line datasets
=============================================================
Sibling of few_shot_hf_ds.py: same dataset, same prompts, same inference path,
same result JSON — but the model is LoRA/QLoRA fine-tuned on the *train* split
first and then evaluated on the *test* split. Results therefore drop straight
next to the ZS / N_Shot numbers of few_shot_hf_ds.py.

Training itself mirrors train_qwen_lora.py and train_gemma_lora.py
(TRL SFTTrainer + PEFT), but one script covers every family instead of one
script per family, because nothing here is written per family:

    * message building — HF datasets hand out PIL images, which both the Qwen
      and the Gemma processors take inline, so one code path covers both.
    * label masking    — the prompt/answer boundary is read off the chat
      template (see make_collate_fn), not off per-family header tokens. Nothing
      to extend when a new model family shows up.
    * modules_to_save  — the one genuine family difference: Gemma ties its
      embeddings, so lm_head + embed_tokens are trained alongside the adapter
      (--modules-to-save auto, overridable).

Data scaling: --limit N draws N training samples with --seed. The draw is one
seeded permutation truncated to N, so with the same seed the N=100 set is a
subset of the N=500 set — the curve measures added data, not a reshuffle.

Only the training split is limited; the test split is always evaluated in full.

Usage:
    python src/lora_train_hf_ds.py \
        --model Qwen/Qwen3.5-4B --dataset deepcopy/LAM-handwritten --limit 500

    python src/lora_train_hf_ds.py \
        --model google/gemma-4-E4B-it --dataset deepcopy/LAM-handwritten \
        --4-bit --limit 2000 --epochs 2

    # data-scaling sweep — same seed, nested training subsets
    for n in 100 250 500 1000 2000; do
        python src/lora_train_hf_ds.py \
            --model Qwen/Qwen3.5-4B --limit $n --seed 3
    done

Requires (fiftyone conda env has these):
    pip install transformers torch accelerate peft trl Pillow tqdm datasets
    pip install bitsandbytes   # only for --4-bit
"""

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# ── Pin to a single GPU BEFORE torch initializes CUDA ──────────────────────
# Prevents HF Trainer from wrapping the model in DataParallel. Override for
# multi-GPU by passing --device-map auto.
if not any(a.split("=")[0] == "--device-map" for a in sys.argv):
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = _cvd.split(",")[0]

import torch  # noqa: E402  -- must follow the CUDA_VISIBLE_DEVICES pin above
from peft import LoraConfig, get_peft_model  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# Prompts, model loading knobs, inference and VRAM accounting come from the
# zero/few-shot scripts so a fine-tuned run stays comparable to them.
from few_shot_hf_ds import clean_prediction, compute_metrics  # noqa: E402
from gpu_stats import (  # noqa: E402
    gpu_devices_info,
    model_vram_info,
    pass_entry,
    print_model_vram,
    reset_peak_vram,
    vram_snapshot,
)
from predict_transformers import (  # noqa: E402
    DTYPE,
    ENABLE_THINKING,
    TEMPERATURE,
    attn_candidates,
    build_chat_messages,
    predict,
)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA / QLoRA fine-tune a VLM for HTR on a HuggingFace line dataset")

    # Data
    p.add_argument("--dataset", default="Teklia/RIMES-2011-line",
                   help="HuggingFace dataset id (default: Teklia/RIMES-2011-line)")
    p.add_argument("--split", default="test", help="Test split name (default: test)")
    p.add_argument("--train-split", default="train", help="Train split name (default: train)")
    p.add_argument("--image-col", default="image", help="Image column name (default: image)")
    p.add_argument("--text-col", default="text", help="Text/label column name (default: text)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of TRAINING samples, drawn with --seed. "
                        "Nested across limits: same seed => the smaller set is a "
                        "subset of the larger. Default: full train split. The test "
                        "split is always evaluated in full.")

    # Model
    p.add_argument("--model", required=True, help="HuggingFace model id (must support vision)")
    p.add_argument("--processor", default=None, help="Processor id (default: --model)")
    p.add_argument("--device", default="cuda", help="CUDA device for generation (default: cuda)")
    p.add_argument("--device-map", default=None,
                   help="Device map for loading. Default: single GPU. 'auto' for multi-GPU.")

    # LoRA / QLoRA
    p.add_argument("--4-bit", dest="four_bit", action="store_true",
                   help="QLoRA: load the base model in NF4 4-bit (bitsandbytes).")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    p.add_argument("--lora-target-modules", default="all-linear",
                   help="Comma-separated module names or 'all-linear' (default: all-linear)")
    p.add_argument("--modules-to-save", default="auto",
                   help="Extra modules trained in full. 'auto' = lm_head,embed_tokens for "
                        "Gemma (tied embeddings), nothing otherwise. 'none' to disable.")

    # Training hyper-params (defaults match train_{qwen,gemma}_lora.py)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", default="cosine", choices=["constant", "cosine", "linear"])
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--optim", default="adamw_torch_fused")
    p.add_argument("--max-length", type=int, default=1024, help="Max sequence length (tokens)")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-strategy", default="no", choices=["no", "epoch"],
                   help="Per-epoch eval loss on the full test split. Off by default — "
                        "test splits here are large and the final scored pass is what counts.")
    p.add_argument("--save-strategy", default="no", choices=["no", "epoch"],
                   help="Intermediate checkpoints (default: no — the adapter is saved at the end).")
    p.add_argument("--report-to", default="tensorboard", choices=["tensorboard", "wandb", "none"])
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="Keep checkpoint-* dirs in the run dir instead of pruning them.")

    # Generation / output
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max new tokens (default: 128). Line transcriptions are short: the "
                        "longest reference in any test split used here is ~34 tokens (RIMES) "
                        "plus ~4 for the 'Transcription:' prefix, so 64 only ever binds on "
                        "degenerate repetition loops. Raise it for longer-line data "
                        "(NorHand's train split reaches 73 tokens).")
    p.add_argument("--prediction-key", default=None,
                   help="Label for the predictions in the result JSON "
                        "(default: 'LoRA' / 'QLoRA').")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results JSON (default: <repo>/results)")
    p.add_argument("--run-dir", default=None,
                   help="Where to write the trained adapter (default: <repo>/runs/"
                        "<dataset>__<model>_<method>_n<N>_e<epochs>_seed<seed>, the same "
                        "stem as the results JSON)")

    # Misc
    p.add_argument("--seed", type=int, default=3,
                   help="Seed for the training subsample and training itself (default: 3)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Dataset — seeded, nested training subsample
# ---------------------------------------------------------------------------

def subsample_train(train_ds, limit: int | None, seed: int):
    """Take `limit` training rows, deterministically. Returns (dataset, indices).

    One seeded permutation truncated to `limit`, so a run with limit=100 trains
    on a subset of the limit=500 run's data (same seed). Indices are sorted so
    the on-disk read order stays sequential.
    """
    if limit is None or limit >= len(train_ds):
        return train_ds, None
    perm = list(range(len(train_ds)))
    random.Random(seed).shuffle(perm)
    indices = sorted(perm[:limit])
    return train_ds.select(indices), indices


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------

def is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def load_base_model(model_id: str, four_bit: bool, device_map):
    """Load the base VLM for training. Returns (model, info).

    Quantization config for QLoRA is the one from train_qwen_lora.py
    (quant_storage set so the 4-bit weights keep the compute dtype).

    Attention implementations are tried in predict_transformers.attn_candidates()
    order (sdpa, then eager), so a fine-tuned run loads the model exactly the way
    the ZS / N_Shot runs it is compared against do — and the reasoning for that
    order lives in one place instead of being restated here.
    """
    load_kwargs: dict = dict(dtype=DTYPE, device_map=device_map, trust_remote_code=True)
    if four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_storage=DTYPE,
        )

    model = None
    attn_used = None
    last_err: Exception | None = None
    for attn in attn_candidates():
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, attn_implementation=attn, **load_kwargs)
            attn_used = attn
            break
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # unsupported attn implementation for this model
            last_err = e
            # A from_pretrained that dies partway can leave weights on the
            # GPU. Dropping them keeps the successful attempt's after_load
            # (and the load peak) from counting memory no live model holds.
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  attn_implementation={attn!r} failed "
                  f"({type(last_err).__name__}: {last_err}), trying next")

    if model is None:
        raise RuntimeError(f"Could not load {model_id}: {last_err}")

    # Same accounting as every other script in the family (gpu_stats), so the
    # vram numbers of a fine-tuned run and a ZS run are the same quantity —
    # including offloaded_GiB, which is the only warning that device_map pushed
    # part of the weights to CPU and made everything 10x slower.
    info = model_vram_info(model)
    info["attn_implementation"] = attn_used
    print_model_vram(info)
    return model, info


def build_peft_config(args) -> LoraConfig:
    """LoRA config; Gemma additionally trains its tied lm_head + embed_tokens."""
    target_modules = (
        args.lora_target_modules
        if args.lora_target_modules == "all-linear"
        else [m.strip() for m in args.lora_target_modules.split(",")]
    )

    if args.modules_to_save == "auto":
        modules_to_save = ["lm_head", "embed_tokens"] if is_gemma(args.model) else None
    elif args.modules_to_save == "none":
        modules_to_save = None
    else:
        modules_to_save = [m.strip() for m in args.modules_to_save.split(",")]

    kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    if modules_to_save:
        # Tied embeddings: the head and the embedding matrix must stay in sync.
        kwargs["modules_to_save"] = modules_to_save
        kwargs["ensure_weight_tying"] = True
    return LoraConfig(**kwargs)


def fix_modules_to_save_devices(model) -> int:
    """Align PEFT's trainable module copies with their original modules' devices.

    Same fix as train_gemma_lora.py: ModulesToSaveWrapper deep-copies a base
    module, and the copy escapes the AlignDevicesHook that dispatch_model()
    installed under device_map='auto' — inputs then arrive on the wrong device.
    """
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from peft.utils.other import ModulesToSaveWrapper

    fixed = 0
    for module in model.modules():
        if not isinstance(module, ModulesToSaveWrapper):
            continue
        try:
            exec_device = next(module.original_module.parameters()).device
        except StopIteration:
            continue
        for copy_mod in module.modules_to_save.values():
            copy_mod.to(exec_device)
        add_hook_to_module(
            module,
            AlignDevicesHook(execution_device=exec_device, io_same_device=False),
        )
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# 4. Chat construction + label masking
# ---------------------------------------------------------------------------

# Not every processor's chat template accepts enable_thinking; probe once.
_SUPPORTS_ENABLE_THINKING = True


def apply_chat_template(processor, messages, **kwargs):
    """Apply the chat template with thinking disabled where supported."""
    global _SUPPORTS_ENABLE_THINKING
    if _SUPPORTS_ENABLE_THINKING:
        try:
            return processor.apply_chat_template(
                messages, enable_thinking=ENABLE_THINKING, **kwargs)
        except (TypeError, ValueError):
            _SUPPORTS_ENABLE_THINKING = False
            print("  note: processor does not accept enable_thinking")
    return processor.apply_chat_template(messages, **kwargs)


# Marker standing in for the answer while the template is rendered. NUL cannot
# occur in a transcription and no template quotes or trims it away.
_ANSWER_SENTINEL = "\x00ANSWER\x00"


def answer_slot(processor, messages) -> tuple[str, str]:
    """Split the training render of `messages` at the assistant's answer.

    Renders the conversation once with a sentinel in place of the answer and
    cuts on it, so what the template puts before and after the answer is read
    off the template rather than guessed. Returns (head, tail).
    """
    rendered = apply_chat_template(
        processor,
        messages + [{"role": "assistant",
                     "content": [{"type": "text", "text": _ANSWER_SENTINEL}]}],
        add_generation_prompt=False, tokenize=False)
    i = rendered.find(_ANSWER_SENTINEL)
    if i < 0:
        raise RuntimeError(
            "Chat template dropped or rewrote the assistant message, so the "
            "answer boundary cannot be located. Check the template for this "
            f"model. Rendered: {rendered[-200:]!r}")
    return rendered[:i], rendered[i + len(_ANSWER_SENTINEL):]


def make_collate_fn(processor, image_col: str, text_col: str):
    """Collator: HF rows → processor batch with everything but the answer masked.

    The training sequence is built as *the generation prompt* — byte-identical
    to what the model is fed at inference — plus the answer and whatever turn
    terminator the template puts after it (answer_slot). Only that suffix is a
    training target; the prompt is masked to -100.

    Composing the sequence this way, instead of rendering the full conversation
    in one go, is what keeps training and inference aligned when a template
    treats the assistant turn differently in the two cases. Gemma 4 is exactly
    that case: with thinking off, its generation prompt ends in an empty thought
    channel (`<|turn>model\\n<|channel>thought\\n<channel|>`), but the same
    template renders a stored assistant turn without one — so the full render is
    not a prefix-extension of the prompt render, and a model trained on it would
    never see at training time the tokens it is asked to continue from at
    inference. Qwen's empty <think></think> pair sits in both renders, so there
    the composed sequence is identical to the full render.

    Nothing here is family-specific: no assistant-header tokens, no role strings.
    """
    tokenizer = processor.tokenizer

    def collate_fn(examples):
        answers, texts, images = [], [], []
        for ex in examples:
            image = ex[image_col].convert("RGB")
            messages = build_chat_messages([], [], target_image=image)
            prompt = apply_chat_template(
                processor, messages, add_generation_prompt=True, tokenize=False)
            head, tail = answer_slot(processor, messages)
            if not prompt.startswith(head):
                raise RuntimeError(
                    "Chat template renders the conversation up to the answer "
                    "differently with and without add_generation_prompt, so the "
                    "answer cannot be appended to the generation prompt. Check "
                    "the template for this model.\n"
                    f"  training render:   ...{head[-120:]!r}\n"
                    f"  generation prompt: ...{prompt[-120:]!r}")
            # .strip() mirrors the trim every template applies to message text.
            answer = (ex[text_col].strip() + tail).rstrip()
            answers.append(answer)
            texts.append(prompt + answer)
            # One image per sample, nested: Gemma's processor rejects a flat list
            # as soon as the batch holds more than one sample.
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Keep only the answer tokens — the tail of the unpadded sequence. Taken
        # from the attention mask, so this is correct for either padding side.
        labels = batch["input_ids"].clone()
        for i, answer in enumerate(answers):
            n_answer = len(tokenizer(answer, add_special_tokens=False).input_ids)
            masked = labels.new_full((labels.size(1),), -100)
            if n_answer > 0:  # nothing to learn from: no answer, no terminator
                keep = batch["attention_mask"][i].nonzero().flatten()[-n_answer:]
                masked[keep] = labels[i, keep]
            labels[i] = masked

        batch["labels"] = labels
        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# 5. Evaluation — same inference path as few_shot_hf_ds.py's ZS pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor, test_ds, ids, image_col, label, predictions,
             device, gen_kwargs, write_fn):
    """Zero-shot generate over the whole test split with the fine-tuned model."""
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    reset_peak_vram()
    t_start = time.time()
    done = 0

    with tqdm(range(len(test_ds)), desc=label, unit="img") as pbar:
        for i in pbar:
            sid = ids[i]
            try:
                target_img = test_ds[i][image_col].convert("RGB")
                raw = predict(model, processor, target_img, [], [], device, gen_kwargs)
                prediction = clean_prediction(raw)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"  FAIL [{label}] {sid}: {e}")
                predictions.setdefault(sid, {})[label] = ""
                continue

            predictions.setdefault(sid, {})[label] = prediction
            done += 1
            pbar.set_postfix_str(prediction[:60])

            if done % 10 == 0:
                write_fn()

    vram = vram_snapshot()
    write_fn()
    return time.time() - t_start, vram


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit("Error: No CUDA device available. This script requires CUDA.")
    device = args.device
    if not (device == "cuda" or device.startswith("cuda:") or device == "auto"):
        sys.exit(f"Error: --device must be a CUDA device (cuda, cuda:0, ...), got: {device}")

    quant = "nf4" if args.four_bit else "none"
    method = "qlora" if args.four_bit else "lora"
    label = args.prediction_key or ("QLoRA" if args.four_bit else "LoRA")
    gen_kwargs: dict = {"max_new_tokens": args.max_tokens, "do_sample": False}

    # --- Dataset splits ---
    from datasets import load_dataset

    print(f"Loading splits: {args.dataset} [{args.train_split} / {args.split}] ...")
    train_ds = load_dataset(args.dataset, split=args.train_split)
    test_ds = load_dataset(args.dataset, split=args.split)
    full_train = len(train_ds)

    for ds, name in ((train_ds, args.train_split), (test_ds, args.split)):
        for col in (args.image_col, args.text_col):
            if col not in ds.column_names:
                sys.exit(f"Error: column {col!r} not in split {name!r} "
                         f"(have: {ds.column_names})")

    train_ds, train_indices = subsample_train(train_ds, args.limit, args.seed)
    n_train, n_total = len(train_ds), len(test_ds)

    # Stable per-sample ids, as in few_shot_hf_ds.py.
    id_col = next((c for c in ("id", "filename", "file_name", "name")
                   if c in test_ds.column_names), None)
    if id_col is not None:
        ids = [str(test_ds[i][id_col]) for i in range(n_total)]
    else:
        ids = [f"{args.split}_{i:05d}" for i in range(n_total)]

    # --- Paths ---
    # Results and adapters land next to the repo, not inside src/ — that folder
    # holds code.
    repo_dir = Path(__file__).resolve().parent.parent
    ds_slug = args.dataset.replace("/", "_").replace("\\", "_").replace(" ", "_")
    model_slug = args.model.replace("/", "_").replace("\\", "_").replace(" ", "_")
    # Epochs are part of the tag: without them a 2-epoch and a 3-epoch run on the
    # same data write the same file, and the second silently replaces the first.
    run_tag = f"{method}_n{n_train}_e{args.epochs}_seed{args.seed}"

    # Adapter dir and results JSON carry the same stem, dataset included: without
    # it two datasets trained with the same model/method/n/seed would write their
    # adapters into one directory and silently overwrite each other.
    run_stem = f"{ds_slug}__{model_slug}_{run_tag}"

    out_dir = Path(args.output_dir) if args.output_dir else repo_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{run_stem}.json"

    run_dir = Path(args.run_dir) if args.run_dir else repo_dir / "runs" / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Model + processor ---
    processor_id = args.processor or args.model
    device_map = args.device_map or {"": 0}
    print(f"Loading {args.model} (processor: {processor_id}) ...")
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
    reset_peak_vram()
    model, model_info = load_base_model(args.model, args.four_bit, device_map)

    peft_config = build_peft_config(args)

    # Multi-GPU + modules_to_save: wrap here so the trainable copies can be put
    # back on their base module's device before SFTTrainer touches the model.
    trainer_peft_config = peft_config
    if args.device_map == "auto" and getattr(peft_config, "modules_to_save", None):
        print("Multi-GPU dispatch: pre-wrapping with PEFT and aligning devices.")
        if args.four_bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, peft_config)
        print(f"  Aligned {fix_modules_to_save_devices(model)} module copies.")
        trainer_peft_config = None

    tok = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    # One config block, mirroring few_shot_hf_ds.py so the two logs line up.
    print(f"\nDataset: {args.dataset} — train {n_train}/{full_train} "
          f"(limit {args.limit}, seed {args.seed}) | test {n_total}")
    print(f"Model: {args.model} — {model_info['n_params_B']}B params, "
          f"{model_info['footprint_GiB']:.2f} GiB")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}) | Dtype: bfloat16 "
          f"| Quant: {quant} | Attn: {model_info['attn_implementation']}")
    print(f"Method: {method.upper()} r={args.lora_r} alpha={args.lora_alpha} "
          f"targets={args.lora_target_modules} "
          f"modules_to_save={getattr(peft_config, 'modules_to_save', None)}")
    print(f"Gen kwargs: {gen_kwargs}")
    print(f"Adapter: {run_dir}")
    print(f"Output:  {output_path}")

    # --- Result writing (called repeatedly during the eval pass) ---
    predictions: dict = {}
    remarks: dict = {}
    for i in range(n_total):
        predictions.setdefault(ids[i], {})["ground_truth"] = test_ds[i][args.text_col]

    def write_fn() -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "model": args.model,
                    "seed": args.seed,
                    "n_shots": 0,
                    "method": method,
                    "train_source": {
                        "split": args.train_split,
                        "n_train": n_train,
                        "limit": args.limit,
                        "indices": train_indices,
                    },
                    "n_samples": n_total,
                    "metrics": {label: compute_metrics(predictions, label)},
                    "predictions": predictions,
                    "remarks": remarks,
                },
                f, ensure_ascii=False, indent=2,
            )

    # --- Training ---
    training_args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        seed=args.seed,
        max_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds if args.eval_strategy != "no" else None,
        peft_config=trainer_peft_config,
        processing_class=processor,
        data_collator=make_collate_fn(processor, args.image_col, args.text_col),
    )

    print(f"\n--- Training ({args.epochs} epochs on {n_train} samples) ---")
    reset_peak_vram()
    t_train = time.time()
    trainer.train()
    train_s = time.time() - t_train
    train_vram = vram_snapshot()

    trainer.save_model()
    if not args.keep_checkpoints:
        for item in run_dir.glob("checkpoint-*"):
            shutil.rmtree(item, ignore_errors=True)
    print(f"Training finished in {train_s:.1f}s — adapter saved.")

    model = trainer.model  # the trained adapter — kept across the teardown below

    # Release the training allocator pool before measuring inference. Peak VRAM
    # is read via reset_peak_memory_stats(), which resets the peak to the
    # *currently reserved* bytes — and the caching allocator holds on to the
    # optimizer/gradient pool long after those tensors are freed. Without this,
    # the eval pass reports roughly the training footprint and is not comparable
    # to the inference-only numbers few_shot_hf_ds.py records.
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del trainer
    for p in model.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()

    # --- Evaluation ---
    print(f"\n--- {label} pass on {n_total} test samples ---")
    infer_s, infer_vram = evaluate(
        model, processor, test_ds, ids, args.image_col, label,
        predictions, device, gen_kwargs, write_fn,
    )

    remarks["config"] = {
        "quant": quant,
        "dtype": "bfloat16",
        "attn_implementation": model_info["attn_implementation"],
        "temperature": TEMPERATURE,
        "do_sample": False,
        "thinking": ENABLE_THINKING,
        "max_new_tokens": args.max_tokens,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "modules_to_save": getattr(peft_config, "modules_to_save", None),
            "adapter_dir": str(run_dir),
        },
    }
    # load_peak_reserved_GiB is mirrored at the top level because
    # evaluate_htr.peak_vram_gib() falls back to it when the pass label is
    # missing from peak_per_pass — same shape as predict_transformers.py writes.
    remarks["gpu"] = {
        "devices": gpu_devices_info(),
        "model": {k: model_info[k]
                  for k in ("n_params_B", "footprint_GiB", "offloaded_GiB",
                            "after_load", "load_peak_reserved_GiB")},
        "load_peak_reserved_GiB": model_info["load_peak_reserved_GiB"],
        "peak_per_pass": {
            "train": pass_entry(train_vram),
            label: pass_entry(infer_vram),
        },
    }
    remarks["runtime"] = {
        "training_s": round(train_s, 1),
        label: {"duration_s": round(infer_s, 1), "images": n_total,
                "s_per_image": round(infer_s / max(n_total, 1), 2),
                "entries_per_second": round(n_total / max(infer_s, 1e-6), 3)},
        "total_s": round(train_s + infer_s, 1),
    }
    write_fn()

    m = compute_metrics(predictions, label)
    print(f"\nDone → {output_path}")
    print(f"  train {train_s:.1f}s ({n_train} samples) | inference {infer_s:.1f}s")
    print(f"  {label:>8}: CER {m['cer_micro']} (micro) / {m['cer_macro']} (macro)  |  "
          f"WER {m['wer_micro']} (micro) / {m['wer_macro']} (macro)  [n={m['n']}]")


if __name__ == "__main__":
    main()
