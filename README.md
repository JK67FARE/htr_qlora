# htr_qlora

Repository for the Paper From Prompting to Adaptation: Resource-Aware HTR with MLLMs

## Setup

```bash
unzip metadata_crops_2000_tvt.zip
pip install -r requirements.txt
```

Python 3.10 or newer. The train/predict scripts need a CUDA GPU and exit without one;
`evaluate_htr.py` and `split_data.py` run on CPU.

Run everything from the repository root:

```bash
python src/evaluate_htr.py --help
```

## Scripts

| Script | What it does |
| --- | --- |
| `split_data.py` | Annotation JSON → stratified train/[validation/]test manifests + image folders |
| `predict_transformers.py` | VLM zero-shot and few-shot on a folder of crops (bf16 or 4-bit NF4) |
| `train_qwen_lora.py` | LoRA/QLoRA fine-tune of a Qwen VLM on the manifest layout |
| `train_gemma_lora.py` | LoRA/QLoRA fine-tune of a Gemma VLM on the manifest layout |
| `train_hf.py` | Full fine-tune of TrOCR on the manifest layout |
| `predict_trocr_hf.py` | Inference with a TrOCR model fine-tuned by `train_hf.py` |
| `few_shot_hf_ds.py` | Zero-shot and few-shot on a HuggingFace line dataset |
| `lora_train_hf_ds.py` | LoRA/QLoRA on a HuggingFace line dataset (Qwen and Gemma) |
| `evaluate_htr.py` | Scores prediction JSONs into a CSV table |
| `gpu_stats.py` | Shared VRAM accounting (library, not a CLI) |

## Typical run

Predict, then score:

```bash
python src/predict_transformers.py \
    --image-dir metadata_crops_2000_tvt/test \
    --model google/gemma-4-E2B-it

python src/evaluate_htr.py \
    --prediction-dir results/ \
    --dataset metadata_crops_2000_tvt \
    --annotations metadata_crops_2000_tvt/test.json
```

Fine-tune, then score the predictions JSON it writes:

```bash
python src/train_qwen_lora.py \
    --model-id Qwen/Qwen3.5-4B \
    --data-dir metadata_crops_2000_tvt \
    --epochs 3

python src/evaluate_htr.py \
    --predictions <output-dir>/<run>_predictions.json \
    --annotations metadata_crops_2000_tvt/test.json
```

The eval tables in `results_csv/` were produced this way.

## Notes

- **`evaluate_htr.py` is the only script that reports a score.** The training scripts
  compute a CER internally, but only to pick the best epoch — they write predictions and
  run metadata, nothing else. Every number in a table comes from one place.
- **Keep a `validation.json`.** Without it the trainers pick their best epoch on the *test*
  split and warn that the reported test CER is optimistic. `split_data.py --val-size 0.15`
  writes one.
- **Seeds.** Prediction files whose names differ only in a `_seed<N>` token are averaged into
  a second `_seedmean.csv` with a standard deviation. Only few-shot varies with the seed; a
  zero-shot row aggregated the same way should show sd 0, which makes it a free sanity check.
- **Try a small model first.** The scripts take `--limit` / `--max-train-samples` to cap the
  training set, which is the quickest way to check a run end to end before committing a GPU
  to it.
- Predictions and adapters are written to `results/` and `runs/`, both gitignored.
