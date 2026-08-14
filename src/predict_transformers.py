"""
VLM HTR Prediction — Transformers, bf16 vs. 4-bit NF4
=====================================================
Loads a VLM locally via HuggingFace transformers and runs HTR over a folder of
images. The output JSON is the schema evaluate_htr.py reads, with one key per
pass, so a run drops straight into an eval table next to every other script here.

Exactly two configurations are supported, so runs stay comparable:

    --quant none   bfloat16 weights (default)
    --quant nf4    4-bit NormalFloat + double quantization, bf16 compute
                   (as in the QLoRA paper)

Decoding is always greedy (temperature 0, do_sample=False) and thinking /
reasoning modes are always disabled.  Images are fed at native resolution.

Requires CUDA.  Runs a zero-shot and a few-shot pass; the few-shot examples are
drawn from a validation-split manifest with a fixed --seed, so the same seed
always yields the same examples.  --fs-only skips the zero-shot pass, which is
what you want when sweeping seeds against an existing ZS baseline.

Usage:
    python src/predict_transformers.py --image-dir metadata_crops_2000_tvt/test \\
        --model Qwen/Qwen3.5-4B
    python src/predict_transformers.py --image-dir metadata_crops_2000_tvt/test \\
        --model Qwen/Qwen3.5-4B--quant nf4
    python src/predict_transformers.py --image-dir metadata_crops_2000_tvt/test \\
        --model Qwen/Qwen3.5-4B --fs-only --seed 7

Requires:
    pip install transformers torch accelerate Pillow tqdm
    pip install bitsandbytes  # only for --quant nf4
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# VRAM accounting shared with the LoRA train/predict scripts, so the vram_gib
# column in one eval table is the same quantity for every source.
from gpu_stats import (
    gpu_devices_info,
    model_vram_info,
    pass_entry,
    print_model_vram,
    reset_peak_vram,
    vram_snapshot,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# System and user prompt adapted from Crosilla et al. 2025
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

# Few-shot examples — drawn from a validation-split manifest. The count is fixed
# so runs stay comparable; only the seed varies.
DEFAULT_VAL_JSON = "metadata_crops_2000_tvt/validation.json"
N_SHOTS = 5

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Fixed decoding config — kept as constants so they show up in remarks.
DTYPE = torch.bfloat16
TEMPERATURE = 0.0
ENABLE_THINKING = False


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks left by models that ignore enable_thinking=False."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

def draw_few_shot(val_json: Path, n_shots: int, seed: int) -> list[tuple[Path, str]]:
    """
    Draw n_shots (image_path, transcription) pairs from a split_data.py JSON.

    Selection depends only on (that JSON, n_shots, seed), so every model
    evaluated with the same seed sees the identical example set.
    """
    with open(val_json, "r", encoding="utf-8") as f:
        blob = json.load(f)

    samples = blob["samples"]

    # split_data.py writes image_dir relative to wherever it was run, so prefer
    # the image folder sitting next to the JSON and fall back to as-written.
    raw_dir = Path(blob.get("image_dir", ""))
    img_dir = next((d for d in (val_json.parent / raw_dir.name, raw_dir) if d.is_dir()),
                   val_json.parent)

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(samples)), min(n_shots, len(samples))))
    return [(img_dir / samples[i]["image_file"], samples[i]["transcription"])
            for i in indices]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_quant_config(quant: str):
    """
    Build a BitsAndBytesConfig for the requested quantization mode.
    Returns None when quant == 'none'.
    """
    if quant == "none":
        return None
    if quant != "nf4":
        raise ValueError(f"Unknown quant mode: {quant!r}")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise RuntimeError(
            f"BitsAndBytesConfig not available in transformers ({e}). "
            "Upgrade transformers and install bitsandbytes."
        )
    try:
        import bitsandbytes  # noqa: F401  -- presence check only
    except ImportError as e:
        raise RuntimeError(
            f"bitsandbytes is required for --quant nf4: pip install bitsandbytes ({e})"
        )

    # QLoRA setup as in the official Gemma QLoRA guide: NF4 + double
    # quantization, bf16 compute and bf16 quant storage. No skip list — the
    # model's own config decides which modules stay unquantized (Gemma keeps
    # its vision tower and lm_head out of the quantized set itself), and a
    # hand-written substring list silently mismatches per architecture.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=DTYPE,
    )


def attn_candidates() -> list[str]:
    """
    Attention implementations to try, best first.

    torch's own SDPA dispatches to the FlashAttention-2 kernel on Ampere and
    newer, so the flash-attn package is not worth its version-matching pain:
    measured on this A6000 (sm_86, torch 2.11), the flash backend runs at
    0.077 ms/call against 0.113 memory-efficient and 1.647 math. sdpa also
    sidesteps FA2's unreliable support for Gemma's SigLIP vision tower.
    eager remains the fallback for models without an sdpa path.
    """
    return ["sdpa", "eager"]


def load_model(model_id: str, device: str, quant: str = "none"):
    """Load model + processor. Returns (model, processor, info)."""

    print(f"Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    quant_config = build_quant_config(quant)
    quantized = quant_config is not None

    if quantized:
        device_map = "auto"
        load_kwargs = dict(
            quantization_config=quant_config,
            dtype=DTYPE,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        # "cuda" (no index) and "auto" both delegate placement to Accelerate
        # so multi-GPU machines actually split the model. Specific devices
        # like "cuda:0" pin to that single device.
        device_map = "auto" if device in ("auto", "cuda") else {"": device}
        load_kwargs = dict(
            dtype=DTYPE,
            device_map=device_map,
            trust_remote_code=True,
        )

    print(f"Loading model from {model_id} → quant={quant}, dtype={DTYPE}, "
          f"device_map={device_map}...")

    # Try each attention implementation in turn, best first.
    model = None
    attn_used = None
    last_err: Optional[Exception] = None
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
            print(f"  attn_implementation={attn!r} failed "
                  f"({type(last_err).__name__}: {last_err}), trying next")

    if model is None:
        raise RuntimeError(f"Could not load {model_id}: {last_err}")

    model.eval()

    info = model_vram_info(model)
    info["attn_implementation"] = attn_used

    print_model_vram(info)
    print(f"  attn={attn_used}")
    for dev, m in info["after_load"].items():
        print(f"  {dev}: {m['allocated_GiB']:.2f} GiB allocated, "
              f"{m['reserved_GiB']:.2f} GiB reserved")

    return model, processor, info


# ---------------------------------------------------------------------------
# Chat template + message building
# ---------------------------------------------------------------------------

# Not every processor's chat template accepts enable_thinking; probe once.
_SUPPORTS_ENABLE_THINKING = True


def _apply_chat_template_inputs(processor, messages: list[dict],
                                add_generation_prompt: bool, device: str):
    """Apply chat template and return a BatchEncoding ready to feed the model."""
    global _SUPPORTS_ENABLE_THINKING

    kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )
    if _SUPPORTS_ENABLE_THINKING:
        try:
            inputs = processor.apply_chat_template(
                messages, enable_thinking=ENABLE_THINKING, **kwargs
            )
            return inputs.to(device)
        except (TypeError, ValueError):
            _SUPPORTS_ENABLE_THINKING = False
            print("  note: processor does not accept enable_thinking, "
                  "relying on <think> stripping instead")

    inputs = processor.apply_chat_template(messages, **kwargs)
    return inputs.to(device)


def build_chat_messages(few_shot_images: list[Image.Image],
                        few_shot_texts: list[str],
                        target_image: Optional[Image.Image] = None) -> list[dict]:
    """Build a chat-format message list with optional few-shot examples."""
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

    for img, transcription in zip(few_shot_images, few_shot_texts):
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": transcription}],
        })

    if target_image is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": target_image},
                {"type": "text", "text": USER_PROMPT},
            ],
        })

    return messages


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, processor, target_image, few_shot_images, few_shot_texts,
            device, gen_kwargs: dict) -> str:
    """Single forward+generate pass for one target image."""
    messages = build_chat_messages(few_shot_images, few_shot_texts, target_image=target_image)
    inputs = _apply_chat_template_inputs(processor, messages, add_generation_prompt=True,
                                         device=device)
    generated = model.generate(**inputs, use_cache=True, **gen_kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return strip_thinking(decoded)


# ---------------------------------------------------------------------------
# Prediction pass
# ---------------------------------------------------------------------------

def write_predictions(path: Path, predictions: dict, image_dir: str, model: str,
                      remarks: dict) -> None:
    """Write the result JSON in the schema evaluate_htr.py reads."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_dir": image_dir,
                "model": model,
                "predictions": predictions,
                "remarks": remarks,
            },
            f, ensure_ascii=False, indent=2,
        )


def run_pass(
    label: str,
    images: list[Path],
    predictions: dict,
    few_shot_images: list[Image.Image],
    few_shot_texts: list[str],
    model, processor,
    device: str,
    gen_kwargs: dict,
    remarks: dict,
    output_path: Path,
    image_dir: str,
    model_id: str,
) -> tuple[float, dict]:
    """
    Run one prediction pass over all images, updating predictions in place.
    Returns (elapsed_seconds, vram_snapshot_with_peaks_for_this_pass).
    """
    t_start = time.time()
    done = 0

    # Peaks are measured per pass: few-shot prompts are much longer, so their
    # activation/KV-cache peak is the interesting one.
    reset_peak_vram()

    with tqdm(images, desc=label, unit="img") as pbar:
        for img_path in pbar:
            fname = img_path.name

            if label in predictions.get(fname, {}):
                continue

            try:
                target_img = Image.open(img_path).convert("RGB")
                prediction = predict(
                    model, processor, target_img,
                    few_shot_images, few_shot_texts,
                    device, gen_kwargs,
                )
            except Exception as e:
                tqdm.write(f"  FAIL [{label}] {fname}: {e}")
                predictions.setdefault(fname, {})[label] = ""
                continue

            predictions.setdefault(fname, {})[label] = prediction
            done += 1
            pbar.set_postfix_str(prediction[:60])

            if done % 10 == 0:
                write_predictions(output_path, predictions, image_dir, model_id, remarks)

    vram = vram_snapshot()
    write_predictions(output_path, predictions, image_dir, model_id, remarks)
    return time.time() - t_start, vram


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VLM HTR with transformers, bf16 or 4-bit NF4 (CUDA required)")
    parser.add_argument("--image-dir", required=True, help="Folder with images to transcribe")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (must support vision)")
    parser.add_argument("--device", default="cuda",
                        help="CUDA device, e.g. cuda, cuda:0, cuda:1 (default: cuda)")
    parser.add_argument("--quant", default="none", choices=["none", "nf4"],
                        help="'none' = bfloat16, 'nf4' = 4-bit NormalFloat + double quant "
                             "with bf16 compute (QLoRA). (default: none)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens (default: 64)")
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON,
                        help=f"split_data.py-style JSON to draw the {N_SHOTS} few-shot examples "
                             f"from (default: {DEFAULT_VAL_JSON})")
    parser.add_argument("--seed", type=int, default=3,
                        help="Seed for the few-shot draw — same seed, same examples (default: 3)")
    parser.add_argument("--fs-only", action="store_true",
                        help="Run only the few-shot pass (skip the zero-shot pass), e.g. when "
                             "sweeping seeds against an already-measured ZS baseline")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already have predictions")
    args = parser.parse_args()

    # CUDA required
    if not torch.cuda.is_available():
        print("Error: No CUDA device available. This script requires CUDA.")
        sys.exit(1)

    device = args.device
    if not (device == "cuda" or device.startswith("cuda:") or device == "auto"):
        print(f"Error: --device must be a CUDA device (cuda, cuda:0, ...), got: {device}")
        sys.exit(1)

    # Greedy decoding, always.
    gen_kwargs: dict = {"max_new_tokens": args.max_tokens, "do_sample": False}

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: {args.image_dir} is not a directory")
        sys.exit(1)

    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    # Load model
    reset_peak_vram()
    model, processor, model_info = load_model(args.model, device, quant=args.quant)

    # Pin pad_token_id to silence generate() warnings
    tok = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    # Load few-shot images — seeded draw from the validation split
    val_json = Path(args.val_json)
    if not val_json.is_file():
        print(f"Error: --val-json not found: {val_json}")
        sys.exit(1)

    print(f"Drawing {N_SHOTS} few-shot examples from {val_json} with seed {args.seed}...")
    fs_images: list[Image.Image] = []
    fs_texts: list[str] = []
    fs_files: list[str] = []
    for p, transcription in draw_few_shot(val_json, N_SHOTS, args.seed):
        if not p.exists():
            print(f"  WARNING: few-shot image not found, skipping: {p}")
            continue
        fs_images.append(Image.open(p).convert("RGB"))
        fs_texts.append(transcription)
        fs_files.append(p.name)
        print(f"  Few-shot loaded: {p.name} → {transcription[:50]}")

    if not fs_images:
        if args.fs_only:
            print("Error: --fs-only was given but no few-shot images could be loaded.")
            sys.exit(1)
        print("WARNING: No few-shot images could be loaded. Few-shot pass will be skipped.")

    n_shots = len(fs_images)
    fs_label = f"{n_shots}_Shot"
    run_zs = not args.fs_only

    # Output filename: model, dataset (last folder of --image-dir), passes, quant
    model_slug = args.model.replace("/", "_").replace("\\", "_").replace(" ", "_")
    dataset_slug = image_dir.resolve().name.replace(" ", "_")
    if not fs_images:
        mode_tag = "_ZS"
    else:
        mode_tag = f"_ZS+{n_shots}shot" if run_zs else f"_{n_shots}shot"
    # Only meaningful when a few-shot pass actually runs.
    seed_tag = f"_seed{args.seed}" if fs_images else ""
    quant_tag = f"_{args.quant}" if args.quant != "none" else "_bf16"
    output_path = Path(
        f"{model_slug}_{dataset_slug}{mode_tag}{seed_tag}{quant_tag}"
        f"_htr_predictions_transformers.json")

    print(f"\nDevice: {device} | Dtype: bfloat16 | Quant: {args.quant} "
          f"| Attn: {model_info['attn_implementation']}")
    print(f"Gen kwargs: {gen_kwargs}")
    print(f"Output: {output_path}")

    # Load predictions
    predictions: dict = {}
    remarks: dict = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            predictions = existing.get("predictions", {})
            remarks = existing.get("remarks", {})
        print(f"Resuming: {len(predictions)} files already have some predictions")

    remarks["config"] = {
        "quant": args.quant,
        "dtype": "bfloat16",
        "attn_implementation": model_info["attn_implementation"],
        "temperature": TEMPERATURE,
        "do_sample": False,
        "thinking": ENABLE_THINKING,
        "max_new_tokens": args.max_tokens,
        "seed": args.seed,
        "few_shot_source": {"val_json": str(val_json), "image_files": fs_files},
    }
    # attn_* live in remarks["config"], so only the VRAM fields go under
    # gpu["model"]. load_peak_reserved_GiB is mirrored at the top level because
    # that is where evaluate_htr.peak_vram_gib() looks when a pass label is
    # missing from peak_per_pass.
    remarks["gpu"] = {
        "devices": gpu_devices_info(),
        "model": {k: model_info[k] for k in
                  ("n_params", "n_params_B", "footprint_GiB", "offloaded_GiB",
                   "after_load", "load_peak_reserved_GiB")},
        "load_peak_reserved_GiB": model_info["load_peak_reserved_GiB"],
        # Preserved across --resume so an interrupted run keeps earlier passes.
        "peak_per_pass": remarks.get("gpu", {}).get("peak_per_pass", {}),
    }
    peak_per_pass = remarks["gpu"]["peak_per_pass"]

    n_total = len(images)
    runtime: dict = remarks.get("runtime", {})
    t_total_start = time.time()

    passes = ([] if not run_zs else ["ZS"]) + ([f"{n_shots}-shot"] if fs_images else [])
    print(f"\n{n_total} images — passes: {' + '.join(passes)}\n")

    def record(label: str, elapsed: float, vram: dict) -> None:
        runtime[label] = {"duration_s": round(elapsed, 1), "images": n_total,
                          "s_per_image": round(elapsed / max(n_total, 1), 2),
                          "entries_per_second": round(n_total / max(elapsed, 1e-6), 3)}
        peak_per_pass[label] = pass_entry(vram)
        print(f"  peak VRAM [{label}]: {peak_per_pass[label]['peak_reserved_total_GiB']:.2f} GiB "
              f"reserved")

    # --- Zero-shot pass ---
    if run_zs:
        print("--- Zero-shot pass ---")
        elapsed, vram = run_pass(
            "ZS", images, predictions,
            [], [],
            model, processor,
            device, gen_kwargs,
            remarks, output_path, str(image_dir), args.model,
        )
        record("ZS", elapsed, vram)
        print()

    # --- Few-shot pass ---
    if fs_images:
        print(f"--- {n_shots}-shot pass ---")
        elapsed, vram = run_pass(
            fs_label, images, predictions,
            fs_images, fs_texts,
            model, processor,
            device, gen_kwargs,
            remarks, output_path, str(image_dir), args.model,
        )
        record(fs_label, elapsed, vram)
        print()

    runtime["total_s"] = round(time.time() - t_total_start, 1)
    remarks["runtime"] = runtime
    write_predictions(output_path, predictions, str(image_dir), args.model, remarks)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
