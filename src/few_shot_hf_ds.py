"""
VLM HTR on HuggingFace line datasets — Transformers
===================================================
Sibling of predict_transformers.py, but instead of a folder of image
crops it evaluates a VLM on the *test* split of a HuggingFace HTR line
dataset (e.g. Teklia/NorHand-v3).

By default two passes are run per model, mirroring predict_transformers.py:
    * ZS       — zero-shot
    * N_Shot   — few-shot, using N examples sampled from the *train* split.
Pass --fs-only to run just the few-shot pass, and --4-bit for NF4 quantization.

The few-shot examples are drawn with a fixed --seed, so switching models
keeps the exact same N example images/transcriptions. Because the dataset
ships ground-truth transcriptions, CER/WER are computed and stored in the
result JSON.

Model loading, prompts, chat construction, single-image inference and VRAM
accounting are imported from predict_transformers.py so both scripts stay in
lock-step. Decoding is fixed there: bf16, greedy (temperature 0), thinking off.
Peak VRAM is recorded per pass under remarks.gpu.peak_per_pass, and the
attention implementation actually used lands in remarks.config.

Usage:
    python src/few_shot_hf_ds.py --model google/gemma-3-4b-it
    python src/few_shot_hf_ds.py --model google/gemma-3-4b-it --fs-only
    python src/few_shot_hf_ds.py --model Qwen/Qwen2-VL-7B-Instruct --4-bit

Requires (fiftyone conda env has these):
    pip install transformers torch accelerate Pillow tqdm datasets
    pip install bitsandbytes   # only for --4-bit
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import editdistance
import torch
from datasets import load_dataset
from tqdm import tqdm

# Everything model/prompt related comes from the sibling script so the two stay
# comparable, and the VRAM accounting from gpu_stats so both report the same
# quantity.
from gpu_stats import gpu_devices_info, pass_entry, reset_peak_vram, vram_snapshot
from predict_transformers import (
    ENABLE_THINKING,
    TEMPERATURE,
    load_model,
    predict,
    strip_thinking,
)


# ---------------------------------------------------------------------------
# Prediction cleanup + metrics
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^\s*Transcription:\s*", re.IGNORECASE)


def clean_prediction(text: str) -> str:
    """Strip <think> blocks and the leading 'Transcription:' label."""
    text = strip_thinking(text)
    text = _PREFIX_RE.sub("", text)
    return text.strip()


def compute_metrics(predictions: dict, label: str) -> dict:
    """Corpus CER/WER over every sample that has a prediction for `label`.

    Reports both micro (sum of edits / sum of reference length) and macro
    (mean of per-sample rates) averages.
    """
    char_edits = char_ref = word_edits = word_ref = 0
    cer_rates: list[float] = []
    wer_rates: list[float] = []

    for entry in predictions.values():
        if label not in entry:
            continue
        ref = entry.get("ground_truth", "")
        hyp = entry[label]

        # characters
        ce = editdistance.eval(ref, hyp)
        char_edits += ce
        char_ref += len(ref)
        cer_rates.append(ce / len(ref) if ref else float(hyp != ""))

        # words
        r_words, h_words = ref.split(), hyp.split()
        we = editdistance.eval(r_words, h_words)
        word_edits += we
        word_ref += len(r_words)
        wer_rates.append(we / len(r_words) if r_words else float(len(h_words) > 0))

    n = len(cer_rates)
    return {
        "n": n,
        "cer_micro": round(char_edits / char_ref, 4) if char_ref else None,
        "wer_micro": round(word_edits / word_ref, 4) if word_ref else None,
        "cer_macro": round(sum(cer_rates) / n, 4) if n else None,
        "wer_macro": round(sum(wer_rates) / n, 4) if n else None,
    }


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def sample_few_shot(dataset: str, train_split: str, image_col: str, text_col: str,
                    n_shots: int, seed: int):
    """Deterministically sample n_shots (image, text) pairs from the train split.

    Selection depends only on (dataset, train_split, seed, n_shots), so every
    model evaluated with the same seed sees the identical example set.
    """
    train = load_dataset(dataset, split=train_split)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(train)), min(n_shots, len(train))))

    fs_images, fs_texts = [], []
    for i in indices:
        row = train[i]
        fs_images.append(row[image_col].convert("RGB"))
        fs_texts.append(row[text_col])
    return fs_images, fs_texts, indices


# ---------------------------------------------------------------------------
# Prediction pass over a HF split
# ---------------------------------------------------------------------------

def run_pass(
    label: str,
    test_ds,
    ids: list[str],
    image_col: str,
    predictions: dict,
    few_shot_images: list,
    few_shot_texts: list,
    model, processor,
    device: str,
    gen_kwargs: dict,
    remarks: dict,
    output_path: Path,
    write_fn,
) -> tuple[float, dict]:
    """
    Run one prediction pass over every test sample, updating predictions in place.
    Returns (elapsed_seconds, vram_snapshot_with_peaks_for_this_pass).
    """
    t_start = time.time()
    done = 0

    # Peaks are measured per pass: few-shot prompts are much longer, so their
    # activation/KV-cache peak is the interesting one.
    reset_peak_vram()

    with tqdm(range(len(test_ds)), desc=label, unit="img") as pbar:
        for i in pbar:
            sid = ids[i]
            if label in predictions.get(sid, {}):
                continue

            try:
                target_img = test_ds[i][image_col].convert("RGB")
                raw = predict(
                    model, processor, target_img,
                    few_shot_images, few_shot_texts,
                    device, gen_kwargs,
                )
                prediction = clean_prediction(raw)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"  FAIL [{label}] {sid}: {e}")
                predictions.setdefault(sid, {})[label] = ""
                continue

            predictions.setdefault(sid, {})[label] = prediction
            done += 1
            pbar.set_postfix_str(prediction[:60])

            if done % 10 == 0:
                write_fn(predictions, remarks)

    vram = vram_snapshot()
    write_fn(predictions, remarks)
    return time.time() - t_start, vram


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VLM HTR on a HuggingFace line dataset test split (CUDA required)")
    parser.add_argument("--dataset", default="Teklia/RIMES-2011-line",
                        help="HuggingFace dataset id (default: Teklia/RIMES-2011-line)")
    parser.add_argument("--model", required=True, help="HuggingFace model id (must support vision)")
    parser.add_argument("--split", default="test", help="Test split name (default: test)")
    parser.add_argument("--train-split", default="train",
                        help="Split to draw few-shot examples from (default: train)")
    parser.add_argument("--image-col", default="image", help="Image column name (default: image)")
    parser.add_argument("--text-col", default="text", help="Text/label column name (default: text)")
    parser.add_argument("--n-shots", type=int, default=5, help="Few-shot examples (default: 5)")
    parser.add_argument("--seed", type=int, default=3,
                        help="Seed for few-shot sampling — same seed => same examples (default: 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of test samples (default: full split)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write results (default: <repo>/results)")
    parser.add_argument("--device", default="cuda", help="CUDA device (default: cuda)")
    parser.add_argument("--fs-only", action="store_true",
                        help="Run only the few-shot pass (skip the zero-shot pass)")
    parser.add_argument("--4-bit", dest="four_bit", action="store_true",
                        help="Load the model with NF4 4-bit bitsandbytes quantization (as in predict_transformers.py)")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have predictions")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens (default: 512)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: No CUDA device available. This script requires CUDA.")
        sys.exit(1)

    device = args.device
    if not (device == "cuda" or device.startswith("cuda:") or device == "auto"):
        print(f"Error: --device must be a CUDA device (cuda, cuda:0, ...), got: {device}")
        sys.exit(1)

    # Fixed to match how predict_transformers.py is invoked: bf16, greedy
    # decoding, thinking off. Quant is off unless --4-bit is passed.
    quant = "nf4" if args.four_bit else "none"
    run_zs = not args.fs_only

    # --- Load dataset splits ---
    print(f"Loading test split: {args.dataset} [{args.split}] ...")
    test_ds = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
    n_total = len(test_ds)

    for col in (args.image_col, args.text_col):
        if col not in test_ds.column_names:
            print(f"Error: column {col!r} not in dataset (have: {test_ds.column_names})")
            sys.exit(1)

    # Stable per-sample ids. Prefer a real id column if the dataset has one.
    id_col = next((c for c in ("id", "filename", "file_name", "name") if c in test_ds.column_names), None)
    if id_col is not None:
        ids = [str(test_ds[i][id_col]) for i in range(n_total)]
    else:
        ids = [f"{args.split}_{i:05d}" for i in range(n_total)]

    # --- Few-shot examples from train split (deterministic via seed) ---
    print(f"Sampling {args.n_shots} few-shot examples from [{args.train_split}] with seed {args.seed} ...")
    fs_images, fs_texts, fs_indices = sample_few_shot(
        args.dataset, args.train_split, args.image_col, args.text_col,
        args.n_shots, args.seed,
    )
    for idx, txt in zip(fs_indices, fs_texts):
        print(f"  Few-shot [{args.train_split}:{idx}] → {txt[:60]}")
    n_shots = len(fs_images)
    fs_label = f"{n_shots}_Shot"

    # --- Load model ---
    reset_peak_vram()
    model, processor, model_info = load_model(args.model, device, quant=quant)

    # Greedy decoding, same as predict_transformers.py. pad_token_id is pinned
    # explicitly to silence generate()'s warning; `is None` rather than `or`,
    # because Gemma's pad id is 0 and would otherwise read as missing.
    gen_kwargs: dict = {"max_new_tokens": args.max_tokens, "do_sample": False}
    tok = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    # --- Output path: <dataset>_<model>...json ---
    # Results land next to the repo, not inside src/ — that folder holds code.
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).resolve().parent.parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_slug = args.dataset.replace("/", "_").replace("\\", "_").replace(" ", "_")
    model_slug = args.model.replace("/", "_").replace("\\", "_").replace(" ", "_")
    mode_tag = f"{n_shots}shot" if args.fs_only else f"ZS+{n_shots}shot"
    quant_tag = "_nf4" if args.four_bit else ""
    output_path = out_dir / f"{ds_slug}__{model_slug}_{mode_tag}_seed{args.seed}{quant_tag}.json"

    def write_fn(predictions: dict, remarks: dict) -> None:
        metrics = {}
        if run_zs:
            metrics["ZS"] = compute_metrics(predictions, "ZS")
        if fs_images:
            metrics[fs_label] = compute_metrics(predictions, fs_label)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "model": args.model,
                    "seed": args.seed,
                    "n_shots": n_shots,
                    "few_shot_source": {"split": args.train_split, "indices": fs_indices},
                    "n_samples": n_total,
                    "metrics": metrics,
                    "predictions": predictions,
                    "remarks": remarks,
                },
                f, ensure_ascii=False, indent=2,
            )

    passes = (["ZS"] if run_zs else []) + ([fs_label] if fs_images else [])
    print(f"\nDataset: {args.dataset} [{args.split}] — {n_total} samples")
    print(f"Device: {device} | Dtype: bfloat16 | Quant: {quant} "
          f"| Attn: {model_info['attn_implementation']} | Passes: {', '.join(passes)}")
    print(f"Gen kwargs: {gen_kwargs}")
    print(f"Output: {output_path}")

    # --- Load / resume predictions ---
    predictions: dict = {}
    remarks: dict = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            predictions = existing.get("predictions", {})
            remarks = existing.get("remarks", {})
        print(f"Resuming: {len(predictions)} samples already have some predictions")

    # Ground truth is always (re)written so metrics have a reference.
    for i in range(n_total):
        predictions.setdefault(ids[i], {})["ground_truth"] = test_ds[i][args.text_col]

    remarks["config"] = {
        "quant": quant,
        "dtype": "bfloat16",
        "quant_storage": "bfloat16" if quant != "none" else None,
        "attn_implementation": model_info["attn_implementation"],
        "temperature": TEMPERATURE,
        "do_sample": False,
        "thinking": ENABLE_THINKING,
        "max_new_tokens": args.max_tokens,
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
        # Preserved across --resume so an interrupted run keeps earlier passes.
        "peak_per_pass": remarks.get("gpu", {}).get("peak_per_pass", {}),
    }
    peak_per_pass = remarks["gpu"]["peak_per_pass"]

    runtime: dict = remarks.get("runtime", {})
    t_total_start = time.time()

    def record(label: str, elapsed: float, vram: dict) -> None:
        runtime[label] = {"duration_s": round(elapsed, 1), "images": n_total,
                          "s_per_image": round(elapsed / max(n_total, 1), 2),
                          "entries_per_second": round(n_total / max(elapsed, 1e-6), 3)}
        peak_per_pass[label] = pass_entry(vram)
        print(f"  peak VRAM [{label}]: "
              f"{peak_per_pass[label]['peak_reserved_total_GiB']:.2f} GiB reserved")

    # --- Zero-shot pass ---
    if run_zs:
        print("\n--- Zero-shot pass ---")
        elapsed, vram = run_pass(
            "ZS", test_ds, ids, args.image_col, predictions,
            [], [], model, processor, device, gen_kwargs,
            remarks, output_path, write_fn,
        )
        record("ZS", elapsed, vram)

    # --- Few-shot pass ---
    if fs_images:
        print(f"\n--- {n_shots}-shot pass ---")
        elapsed, vram = run_pass(
            fs_label, test_ds, ids, args.image_col, predictions,
            fs_images, fs_texts, model, processor, device, gen_kwargs,
            remarks, output_path, write_fn,
        )
        record(fs_label, elapsed, vram)

    runtime["total_s"] = round(time.time() - t_total_start, 1)
    remarks["runtime"] = runtime
    write_fn(predictions, remarks)

    # Final metric summary to stdout
    print(f"\nDone → {output_path}")
    final = {}
    if run_zs:
        final["ZS"] = compute_metrics(predictions, "ZS")
    if fs_images:
        final[fs_label] = compute_metrics(predictions, fs_label)
    for lbl, m in final.items():
        print(f"  {lbl:>8}: CER {m['cer_micro']} (micro) / {m['cer_macro']} (macro)  |  "
              f"WER {m['wer_micro']} (micro) / {m['wer_macro']} (macro)  [n={m['n']}]")


if __name__ == "__main__":
    main()
