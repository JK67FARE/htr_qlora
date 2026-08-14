"""
Fine-tuned TrOCR HTR Prediction — folder of images
==================================================
Loads a TrOCR model fine-tuned by train_hf.py and transcribes every image in a
folder. Nothing is configurable beyond the paths: dtype, generation settings
and special tokens all come from the saved checkpoint, so results are exactly
what the fine-tune produces — no inference-time tuning.

The output JSON uses the schema predict_transformers.py writes, with the single
pass labelled "ZS", so it drops straight into evaluate_htr.py:

    { "image_dir": ..., "model": ..., "scale_factor": 1.0,
      "predictions": { "<file.png>": { "ZS": "..." }, ... },
      "remarks": { "gpu": ..., "runtime": ... } }

This script only predicts — no CER/WER. Score it afterwards with:

    python src/evaluate_htr.py \
        --annotations metadata_crops_2000_tvt/test.json \
        --predictions trocr-large-handwritten_ft_htr_predictions_trocr.json \
        --pass ZS --output eval_trocr_ft.csv

Usage:
    python src/predict_trocr_hf.py \
        --model trocr-large-handwritten_ft \
        --image-dir metadata_crops_2000_tvt/test
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Folder/IO helpers from the VLM script, so both write the same JSON structure,
# and the shared VRAM accounting, so both report the same quantity.
from gpu_stats import new_gpu_info, pass_entry, print_model_vram, reset_peak_vram, vram_snapshot
from predict_transformers import IMAGE_EXTS, write_predictions

LABEL = "ZS"


@torch.no_grad()
def transcribe_batch(model, processor, paths: list[Path]) -> list[str]:
    """Transcribe one batch of image paths."""
    images = [Image.open(p).convert("RGB") for p in paths]
    pixel_values = processor.image_processor(
        images=images, return_tensors="pt"
    ).pixel_values.to(model.device)

    generated_ids = model.generate(pixel_values)
    decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in decoded]


def main():
    parser = argparse.ArgumentParser(
        description="HTR inference with a fine-tuned TrOCR model"
    )
    parser.add_argument("--model", required=True,
                        help="Fine-tuned model dir, e.g. trocr-large-handwritten_ft")
    parser.add_argument("--image-dir", required=True, help="Folder with images to transcribe")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: <model-dir>_htr_predictions_trocr.json)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Images per generate() call (default: 16)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    folder_name = image_dir.name if image_dir.is_dir() else image_dir.stem
    output_path = Path(args.output) if args.output else \
        Path(f"trocr_large_{Path(args.model).name}_htr_predictions_{folder_name}.json")

    # Everything — dtype, decoder start/pad/eos ids, max_length, num_beams —
    # comes from the checkpoint's own config and generation_config.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()

    print(f"Model: {args.model} → {device} ({model.dtype})")
    print(f"Generation: max_length={model.generation_config.max_length}, "
          f"num_beams={model.generation_config.num_beams}")
    print(f"Images: {len(images)} from {image_dir}")
    print(f"Output: {output_path}\n")

    predictions: dict = {}
    gpu_info = new_gpu_info(model)
    print_model_vram(gpu_info["model"])
    # Peaks from here on cover inference only, not checkpoint loading.
    reset_peak_vram()
    t_start = time.time()

    for start in tqdm(range(0, len(images), args.batch_size), desc=LABEL, unit="batch"):
        batch = images[start:start + args.batch_size]
        for path, pred in zip(batch, transcribe_batch(model, processor, batch)):
            predictions[path.name] = {LABEL: pred}

    elapsed = time.time() - t_start
    vram = vram_snapshot()
    n = len(images)
    gpu_info["peak_per_pass"][LABEL] = pass_entry(vram)
    remarks = {
        "gpu": gpu_info,
        "runtime": {
            LABEL: {
                "duration_s": round(elapsed, 1),
                "images": n,
                "s_per_image": round(elapsed / max(n, 1), 2),
                "entries_per_second": round(n / max(elapsed, 1e-6), 3),
            },
            "total_s": round(elapsed, 1),
        },
    }
    write_predictions(output_path, predictions, str(image_dir), args.model, remarks)

    print(f"\n{n} images in {elapsed:.1f}s ({elapsed / max(n, 1):.2f}s / image)")
    print(f"peak VRAM [{LABEL}]: "
          f"{gpu_info['peak_per_pass'][LABEL]['peak_reserved_total_GiB']:.2f} GiB reserved")
    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
