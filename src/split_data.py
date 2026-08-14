"""
split_data.py
-------------
Reads the annotation JSON, filters out flagged_unclear images,
strips the "Transcription: " prefix, and creates a train/test split
stratified by the `complicated` flag.

Pass --val-size to carve a validation split out of the train portion as well.
That is the layout the training scripts expect: they pick the best epoch on
validation.json and fall back to the *test* split when it is missing, which
makes the reported test CER optimistic.

Each manifest written here is:
    { "image_dir": "train", "num_samples": N,
      "samples": [{"image_file": "...", "transcription": "...",
                   "complicated": bool, "needs_rotation": bool}, ...] }

`image_dir` is the folder's own name, so the manifest stays valid wherever the
dataset directory is moved to.

Outputs:  train.json  /  [validation.json]  /  test.json

Usage:
    python src/split_data.py --annotations annotations.json --val-size 0.15
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def clean_transcription(text: str) -> str:
    """Remove leading 'Transcription: ' (case-insensitive) if present."""
    return re.sub(r"^Transcription:\s*", "", text, flags=re.IGNORECASE).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Split HTR annotations into train/[validation/]test.")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to the annotation JSON file.")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Folder containing source images (overrides image_dir in JSON).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write the split manifests and image folders.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of all samples held out for test (default 0.2).")
    parser.add_argument("--val-size", type=float, default=0.0,
                        help="Fraction of all samples held out for validation, taken from "
                             "the train portion. 0 (default) writes no validation.json — "
                             "but the training scripts then pick their best epoch on the "
                             "TEST split, which makes the reported test CER optimistic.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    if not 0 <= args.val_size < 1 - args.test_size:
        parser.error(f"--val-size must be in [0, 1 - test_size); got {args.val_size} "
                     f"with --test-size {args.test_size}")

    # ── Load ──────────────────────────────────────────────────────────
    with open(args.annotations, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_dir = Path(args.image_dir) if args.image_dir else Path(data.get("image_dir", ""))
    annotations = data["annotations"]

    # ── Filter & clean ────────────────────────────────────────────────
    samples = []
    for key, ann in annotations.items():
        if ann.get("flagged_unclear", False):
            continue  # skip unclear
        samples.append({
            "image_file": ann["image_file"],
            "transcription": clean_transcription(ann["transcription"]),
            "complicated": ann.get("complicated", False),
            "needs_rotation": ann.get("needs_rotation", False),
        })

    # ── Stratified split ──────────────────────────────────────────────
    # Test comes off the whole set first, then validation off what is left, so
    # --val-size and --test-size are both fractions of the full dataset and the
    # test split is unaffected by whether a validation split was asked for.
    def stratify_on(split_samples):
        return [int(s["complicated"]) for s in split_samples]

    train_samples, test_samples = train_test_split(
        samples,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_on(samples),
    )

    val_samples = []
    if args.val_size > 0:
        train_samples, val_samples = train_test_split(
            train_samples,
            test_size=args.val_size / (1 - args.test_size),
            random_state=args.seed,
            shuffle=True,
            stratify=stratify_on(train_samples),
        )

    # ── Summary ───────────────────────────────────────────────────────
    n_total = len(annotations)
    n_filtered = n_total - len(samples)

    def n_complicated(split_samples):
        return sum(1 for s in split_samples if s["complicated"])

    print(f"Total annotations:     {n_total}")
    print(f"Removed (unclear):     {n_filtered}")
    print(f"Usable samples:        {len(samples)}")
    print(f"Train:                 {len(train_samples)}  "
          f"(complicated: {n_complicated(train_samples)})")
    if val_samples:
        print(f"Validation:            {len(val_samples)}  "
              f"(complicated: {n_complicated(val_samples)})")
    print(f"Test:                  {len(test_samples)}  "
          f"(complicated: {n_complicated(test_samples)})")

    # ── Write ─────────────────────────────────────────────────────────
    annotation_stem = Path(args.annotations).stem
    out = Path(args.output_dir) / annotation_stem
    out.mkdir(parents=True, exist_ok=True)

    def write_split(json_path, img_subdir, split_samples):
        img_subdir.mkdir(parents=True, exist_ok=True)
        copied, missing = 0, 0
        for s in split_samples:
            src = image_dir / s["image_file"]
            if src.exists():
                shutil.copy2(src, img_subdir / src.name)
                copied += 1
            else:
                print(f"  WARNING: image not found: {src}")
                missing += 1
        blob = {
            # The folder's own name, not a path: the manifest sits next to it,
            # so the dataset directory stays valid wherever it is moved to.
            "image_dir": img_subdir.name,
            "num_samples": len(split_samples),
            "samples": split_samples,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2, ensure_ascii=False)
        print(f"Wrote {json_path}  (images copied: {copied}, missing: {missing})")

    write_split(out / "train.json", out / "train", train_samples)
    if val_samples:
        write_split(out / "validation.json", out / "validation", val_samples)
    write_split(out / "test.json", out / "test", test_samples)


if __name__ == "__main__":
    main()
