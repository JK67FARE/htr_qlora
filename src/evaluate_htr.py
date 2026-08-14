"""
HTR Evaluation Script
=====================
Scores HTR prediction JSONs against their references and writes one summary
CSV: a row per (prediction file × pass) with CER/WER/accuracy per normalisation
variant and the run's peak VRAM and throughput. Files that differ only in a
_seed<N> token are additionally averaged into a second CSV, mean ± sd.

Usage:
    # every prediction JSON in a folder whose filename contains the dataset name
    python src/evaluate_htr.py \\
        --prediction-dir results/ \\
        --dataset metadata_crops_2000_tvt \\
        --annotations metadata_crops_2000_tvt/test.json \\
        --pass ZS 5_Shot

    # benchmark dataset — ground truth sits inside the prediction JSONs
    python src/evaluate_htr.py --prediction-dir results/ --dataset LAM

    # explicit file list still works
    python src/evaluate_htr.py --predictions modelA.json modelB.json \\
        --annotations metadata_crops_2000_tvt/test.json

JSON formats expected
---------------------
Annotations — the hand-corrected references for a dataset:
    { ..., "<filename.png>": { "transcription": "...", "flagged_unclear": bool, ... }, ... }

Predictions — the records live under "predictions" (or at the top level) and
each value is either the prediction string itself or a dict of passes:
  Single pass:  { "model": "...", "predictions": { "<filename.png>": "...", ... } }
  Multi-pass:   { "model": "...", "predictions": { "<filename.png>": { "ZS": "...", "5_Shot": "..." }, ... } }
  Multi-pass with built-in ground truth (benchmark datasets):
                { "test_02217": { "ground_truth": "...", "ZS": "...", "5_Shot": "..." }, ... }

  A "ground_truth" key is never treated as a pass; it is the reference for that
  item. --annotations is therefore only needed for datasets whose predictions
  do not carry their own ground truth, and it takes precedence where both are
  present. Use --pass ZS / --pass 5_Shot to select a pass; defaults to all.

Every (file × pass key) is scored as a separate entity, so the ZS and 5_Shot
passes of one run appear as two independent rows.

Metrics  (corpus-level aggregates, per normalisation variant)
------------------------------------------------------------
CER     =  Σ edit_distance(pred_i, ref_i)      /  Σ len(ref_i)
WER     =  Σ word_edit_distance(pred_i, ref_i) /  Σ n_words(ref_i)
acc     =  share of items scoring CER == 0   (transcribed completely correctly)

A "word" is any whitespace-delimited token after normalisation. Edit distance
comes from the `editdistance` package.

Two cost columns come from the prediction file's own "remarks", when the run
recorded them: vram_gib (peak reserved VRAM, summed over devices) and
items_per_s (images per second). Runs without those stats leave them blank.

Normalisation variants
----------------------
Following Romein, Rabus, Leifert & Ströbel (2025), "Assessing advanced
handwritten text recognition engines for digitizing historical documents",
§4.1 and Table 2. Each variant is applied to reference and prediction alike:

    base                Unicode NFC, line breaks → space, whitespace collapsed.
                        Case- and punctuation-sensitive (criteria A and B).
    alnum               keep only Unicode categories L, N, Z, M — drops
                        punctuation and symbols (criterion C).
    alnumlower          alnum + lowercased.
    alnumlower_nospace  own addition: alnumlower with all separators removed,
                        so word separation errors cannot hide (criterion E).

Predictions are cleaned first as we use the Crosilla et al. 2025 prompt:
everything up to and including a leading "Transcription:" label is removed,
since the prompt asks for exactly that label — any other extra output is
scored as an error.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import editdistance
import numpy as np


# ── Normalisation ──────────────────────────────────────────────────────────────
#
# Applied to "in Paris (it's a capital)." the variants give:
#   base                "in Paris (it's a capital)."
#   alnum               "in Paris its a capital"
#   alnumlower          "in paris its a capital"
#   alnumlower_nospace  "inparisitsacapital"

def normalise_base(text: str) -> str:
    """
    Paper's `base`: Unicode NFC, line breaks → space, runs of whitespace
    collapsed, trimmed. Case and punctuation are preserved.
    """
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def _keep_alnum(text: str) -> str:
    """
    Keep only Unicode categories L (letters), N (numbers), Z (separators) and
    M (marks) — i.e. drop punctuation (P), symbols (S) and control chars (C).
    Whitespace is re-collapsed afterwards so that removing " - " does not leave
    a double space that would score as an error.
    """
    kept = ''.join(ch for ch in text if unicodedata.category(ch)[0] in 'LNZM')
    return re.sub(r'\s+', ' ', kept).strip()


def _drop_space(text: str) -> str:
    """Remove every Z-category character (the paper's `nospace`)."""
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'Z')


# Ordered: CER/WER/accuracy are computed once per variant.
VARIANTS: dict[str, Callable[[str], str]] = {
    'base':               normalise_base,
    'alnum':              lambda t: _keep_alnum(normalise_base(t)),
    'alnumlower':         lambda t: _keep_alnum(normalise_base(t)).lower(),
    'alnumlower_nospace': lambda t: _drop_space(_keep_alnum(normalise_base(t)).lower()),
}


# ── Prediction clean-up ────────────────────────────────────────────────────────

# The prompt asks the model for "Transcription:", so that label — and any
# lead-in it puts before it ("Guesses Transcription:") — is forgiven, along
# with the single space that usually follows. Everything else the model emits
# is output it was not asked for and is scored as an error.
#
# Non-greedy: only the FIRST label is consumed. A model that answers twice
# ("Transcription: Welt  Transcription: Zeit") keeps its second guess in the
# string and is penalised for it.
_PREAMBLE = re.compile(r'^.*?Transcription:[ ]?', re.IGNORECASE | re.DOTALL)


def strip_preamble(text: str) -> str:
    """Drop everything up to and including a leading 'Transcription:' label."""
    return _PREAMBLE.sub('', text.strip(), count=1).strip()


# ── JSON loaders ──────────────────────────────────────────────────────────────

_ANNOTATION_META_KEYS = frozenset({'schema_version', 'created', 'image_dir', 'total_images',
                                    'annotator', 'last_modified', 'remarks'})


def load_annotations(path: Path) -> dict:
    """
    Return {filename: record_dict}.

    Handles three schemas:
    - Samples list: {"samples": [{"image_file": "...", "transcription": "..."}, ...], ...}
    - Nested:       {"annotations": {"fname.png": {...}, ...}, ...metadata...}
    - Flat:         {"fname.png": {...}, ...metadata...}
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data.get('samples'), list):
        result = {
            s['image_file']: s
            for s in data['samples']
            if isinstance(s, dict) and 'image_file' in s and 'transcription' in s
        }
    else:
        records = data.get('annotations', data)
        result: dict = {}
        for k, v in records.items():
            if not isinstance(v, dict):
                continue
            if k in _ANNOTATION_META_KEYS:
                continue
            if 'image_file' in v:
                result[v['image_file']] = v
            elif 'transcription' in v:
                result[k] = {**v, 'image_file': k}

    if not result:
        print("  WARNING: no annotation records found — check JSON schema", file=sys.stderr)
        sample = list(data.keys())[:5]
        print(f"  Top-level keys in file: {sample}", file=sys.stderr)
    return result


# Benchmark datasets ship the reference alongside the passes, under this key.
_GT_KEY = 'ground_truth'


class PredictionFile(NamedTuple):
    passes: dict[str, dict]       # {pass_label: {fname: prediction}}
    model: str
    ground_truth: dict[str, str]  # empty unless the file carries its own refs
    remarks: dict                 # run metadata (gpu, runtime, …)


def load_predictions(path: Path) -> PredictionFile:
    """
    Return a PredictionFile; unpacks as (passes, model, ground_truth, remarks).

    Records are read from data["predictions"], or — for benchmark exports
    without that wrapper — from every top-level entry that carries a
    "ground_truth". That key is the reference for the item and is never scored
    as a pass, so `ground_truth` stays empty for our own datasets.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    raw = data.get('predictions')
    if raw is None:
        raw = {k: v for k, v in data.items()
               if isinstance(v, dict) and _GT_KEY in v}

    passes: dict[str, dict] = {}
    ground_truth: dict[str, str] = {}

    for fname, record in raw.items():
        if isinstance(record, dict):
            for label, text in record.items():
                if label == _GT_KEY:
                    ground_truth[fname] = text or ''
                else:
                    passes.setdefault(label, {})[fname] = text or ''
        else:
            passes.setdefault('prediction', {})[fname] = record or ''

    return PredictionFile(passes, data.get('model', path.stem),
                          ground_truth, data.get('remarks', {}))


def annotations_from_ground_truth(ground_truth: dict[str, str]) -> dict:
    """Wrap embedded ground truth in the record shape evaluate() expects."""
    return {fname: {'image_file': fname, 'transcription': gt}
            for fname, gt in ground_truth.items()}


def discover_prediction_files(pred_dir: Path, dataset: str | None) -> list[Path]:
    """Every *.json in *pred_dir* whose filename contains the dataset string."""
    return sorted(p for p in pred_dir.glob('*.json')
                  if dataset is None or dataset in p.name)


# ── Run cost: VRAM and throughput ─────────────────────────────────────────────
#
# Both come from remarks[], which is read in two shapes:
#   current (gpu_stats.py)   gpu = {"devices": …, "load_peak_reserved_GiB": x,
#                                   "peak_per_pass": {"ZS": {"peak_reserved_total_GiB": y}}}
#   older prediction JSONs   gpu = {"cuda:0": {"reserved_GiB": …}, …} plus a
#                            gpu_after_<pass> snapshot per pass
# Peak *reserved* is the headline VRAM number: what the run had to be given.
# Runs recorded without GPU/timing stats yield None.

def _sum_reserved(devices: dict) -> float | None:
    vals = [d.get('peak_reserved_GiB', d.get('reserved_GiB'))
            for d in devices.values() if isinstance(d, dict)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals), 2) if vals else None


def peak_vram_gib(remarks: dict, pass_label: str) -> float | None:
    """Peak reserved VRAM in GiB for *pass_label*, summed over devices."""
    gpu = remarks.get('gpu') or {}

    per_pass = gpu.get('peak_per_pass', {}).get(pass_label, {})
    if 'peak_reserved_total_GiB' in per_pass:
        return per_pass['peak_reserved_total_GiB']
    if per_pass.get('devices'):
        return _sum_reserved(per_pass['devices'])

    after = remarks.get(f'gpu_after_{pass_label}')
    if isinstance(after, dict):
        return _sum_reserved(after)

    # No per-pass figure: fall back to whatever the run recorded overall.
    if 'load_peak_reserved_GiB' in gpu:
        return gpu['load_peak_reserved_GiB']
    return _sum_reserved(gpu.get('devices', gpu))


def items_per_second(remarks: dict, pass_label: str) -> float | None:
    """Inference throughput for *pass_label*, in images per second."""
    run = remarks.get('runtime', {}).get(pass_label)
    if not isinstance(run, dict):
        return None
    if isinstance(run.get('entries_per_second'), (int, float)):
        return round(run['entries_per_second'], 3)
    images, duration = run.get('images'), run.get('duration_s')
    if isinstance(images, (int, float)) and duration:
        return round(images / duration, 3)
    if run.get('s_per_image'):
        return round(1 / run['s_per_image'], 3)
    return None


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(annotations: dict, predictions: dict) -> tuple[list[dict], int]:
    """
    Returns (per_image_results, n_skipped).

    Each result stores raw edit-distance counts and reference lengths *per
    normalisation variant*, so corpus-level CER/WER can be aggregated by
    summing numerators and denominators separately.
    """
    results = []
    n_skipped = 0

    for fname, ann in sorted(annotations.items()):
        if ann.get('flagged_unclear', False):
            n_skipped += 1
            continue
        if fname not in predictions:
            n_skipped += 1
            continue

        transcription = ann.get('transcription')
        if not isinstance(transcription, str):
            print(f"  WARNING: no transcription for {fname} — skipping", file=sys.stderr)
            n_skipped += 1
            continue

        raw_pred = strip_preamble(predictions[fname])

        # An empty reference makes CER and accuracy undefined.
        if not normalise_base(transcription):
            n_skipped += 1
            continue

        per_variant = {}
        for vname, fn in VARIANTS.items():
            ref, pred = fn(transcription), fn(raw_pred)
            ref_tok, hyp_tok = ref.split(), pred.split()
            char_err = editdistance.eval(pred, ref)
            word_err = editdistance.eval(hyp_tok, ref_tok)
            per_variant[vname] = {
                'ref_chars': len(ref),
                'ref_words': len(ref_tok),
                'char_err':  char_err,
                'word_err':  word_err,
                'cer':       char_err / len(ref) if ref else 0.0,
                'wer':       word_err / len(ref_tok) if ref_tok else 0.0,
                'correct':   char_err == 0,
            }

        results.append({
            'filename':    fname,
            'complicated': bool(ann.get('complicated', False)),
            'variants':    per_variant,
        })

    return results, n_skipped


# ── Summary table ─────────────────────────────────────────────────────────────

def _suffix(vname: str) -> str:
    """base → '' (plain 'cer'), others → '_alnum' etc."""
    return '' if vname == 'base' else f'_{vname}'


def _aggregate(subset: list[dict]) -> dict:
    """
    Corpus-level CER/WER and exact-match accuracy for every normalisation
    variant:

        cer = Σ edit_distance / Σ reference characters
        wer = Σ word edits     / Σ reference words
        acc = share of items with CER == 0   (completely correct)
    """
    n = len(subset)
    if n == 0:
        return {}

    def _ratio(num, den) -> float:
        return num / den if den else float('nan')

    out = {'n': n}
    for vname in VARIANTS:
        vs = [r['variants'][vname] for r in subset]
        s = _suffix(vname)
        total_chars = sum(v['ref_chars'] for v in vs)
        total_words = sum(v['ref_words'] for v in vs)
        char_err = sum(v['char_err'] for v in vs)
        word_err = sum(v['word_err'] for v in vs)
        out.update({
            f'cer{s}':         _ratio(char_err, total_chars),
            f'wer{s}':         _ratio(word_err, total_words),
            f'acc{s}':         sum(v['correct'] for v in vs) / n,
            f'mean_cer{s}':    float(np.mean([v['cer'] for v in vs])),
            f'cer_std{s}':     float(np.std([v['cer'] for v in vs])),
            f'char_err{s}':    char_err,
            f'word_err{s}':    word_err,
            f'total_chars{s}': total_chars,
            f'total_words{s}': total_words,
        })
    return out


def _variant_metric_rows() -> list[tuple[str, str]]:
    """(label, key) pairs — base first with the legacy labels intact."""
    rows = [('CER (corpus)',          'cer'),
            ('WER (corpus)',          'wer'),
            ('Accuracy (exact)',      'acc'),
            ('CER mean (per-sample)', 'mean_cer'),
            ('CER std (per-sample)',  'cer_std')]
    for vname in VARIANTS:
        if vname == 'base':
            continue
        s = _suffix(vname)
        rows += [(f'CER {vname}',      f'cer{s}'),
                 (f'WER {vname}',      f'wer{s}'),
                 (f'Accuracy {vname}', f'acc{s}')]
    return rows + [('Char errors',     'char_err'),
                   ('Word errors',     'word_err'),
                   ('Ref chars total', 'total_chars'),
                   ('Ref words total', 'total_words'),
                   ('N samples',       'n')]


_METRIC_ROWS = _variant_metric_rows()


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def build_comparison_report(pass_results: dict[str, list[dict]],
                            n_skipped: int) -> tuple[list[dict], list[str]]:
    """
    Long report as (rows, fieldnames): one row per metric, one column per pass,
    in three blocks — All / Non-complicated / Complicated.
    """
    passes = list(pass_results)
    fields = ['group', 'metric'] + passes
    blank = {p: '' for p in passes}
    rows: list[dict] = []

    def _add_group(label: str, keep) -> None:
        aggs = {p: _aggregate([r for r in res if keep(r)])
                for p, res in pass_results.items()}
        if not any(aggs.values()):
            return
        for i, (name, key) in enumerate(_METRIC_ROWS):
            rows.append({'group': label if i == 0 else '', 'metric': name,
                         **{p: _fmt(a[key]) if a else '' for p, a in aggs.items()}})
        rows.append({'group': '', 'metric': '', **blank})

    _add_group('All',             lambda r: True)
    _add_group('Non-complicated', lambda r: not r['complicated'])
    _add_group('Complicated',     lambda r: r['complicated'])

    n_evaluated = len(next(iter(pass_results.values())))
    rows.append({'group': 'Dataset', 'metric': 'N evaluated', **blank, passes[0]: n_evaluated})
    rows.append({'group': '',        'metric': 'N skipped',   **blank, passes[0]: n_skipped})

    return rows, fields


def build_report(results: list[dict], n_skipped: int) -> list[dict]:
    """Single-pass long report, with columns: group, metric, value."""
    rows, _ = build_comparison_report({'value': results}, n_skipped)
    return rows


# ── Wide table across many prediction files ───────────────────────────────────

_PRED_SUFFIX = re.compile(r'_(htr_)?predictions(_.*)?$')

# Metrics repeated per normalisation variant.
_VARIANT_METRICS = ('cer', 'wer', 'acc')

# Columns of the multi-file table, in order.
_MATRIX_COLS = (['source', 'model', 'pass', 'n', 'total_chars',
                 'vram_gib', 'items_per_s']
                + [f'{m}{_suffix(v)}' for v in VARIANTS for m in _VARIANT_METRICS])


def _strip_pred_suffix(path: Path) -> str:
    """'Qwen3.6-27B_lora_predictions.json' → 'Qwen3.6-27B_lora'."""
    return _PRED_SUFFIX.sub('', path.stem) or path.stem


def source_id(path: Path, pass_label: str) -> str:
    """
    'Qwen3.6-27B_lora_predictions.json' + 'ZS' → 'Qwen3.6-27B_lora_ZS'.
    One id per (run × pass), so a row here names exactly one measured entity.
    """
    return f'{_strip_pred_suffix(path)}_{pass_label}'


# Quantisation is a property of the run, not of the pass, but it only shows up
# in the filename — so a "_nf4" file marks its passes as 4-bit in the table.
_NF4_RE = re.compile(r'_nf4', re.IGNORECASE)


def pass_display(path: Path, pass_label: str) -> str:
    """'ZS' → 'ZS NF4' when the prediction file's name carries a _nf4 token."""
    return f'{pass_label} NF4' if _NF4_RE.search(path.name) else pass_label


class EvaluatedSource(NamedTuple):
    source_id: str
    model: str
    pass_label: str
    results: list[dict]
    vram_gib: float | None = None
    items_per_s: float | None = None


def build_matrix_rows(evaluated: list[EvaluatedSource]) -> list[dict]:
    """
    *evaluated* is [(source_id, model, pass_label, results, vram, items/s), …].
    Returns one flat row per source, ready for csv.DictWriter(_MATRIX_COLS).
    """
    rows = []
    for sid, model, pass_label, results, vram, speed in evaluated:
        a = _aggregate(results)
        if not a:
            continue
        row = {'source': sid, 'model': model, 'pass': pass_label,
               'n': a['n'], 'total_chars': a['total_chars'],
               'vram_gib': '' if vram is None else vram,
               'items_per_s': '' if speed is None else speed}
        for v in VARIANTS:
            s = _suffix(v)
            row.update({f'{m}{s}': a[f'{m}{s}'] for m in _VARIANT_METRICS})
        rows.append(row)
    return rows


# ── Averaging repeated runs of the same config over seeds ─────────────────────

_SEED_RE = re.compile(r'_seed(\d+)')
# A --fs-only rerun of the same config also drops the "ZS+" from its mode tag
# (few_shot_hf_ds.py builds it as "ZS+5shot" or "5shot"). Which passes a file
# happens to hold does not change what a row measures — the pass is its own
# column, and it carries the shot count — so the tag goes too.
_MODE_RE = re.compile(r'_(ZS\+)?\d+shot')


def _seed_group(source: str) -> str:
    """Source id with the seed and mode tokens removed — the grouping key."""
    return _MODE_RE.sub('', _SEED_RE.sub('', source))


# Everything numeric is averaged; the metrics additionally get a "_sd" twin.
_SEED_METRIC_COLS = [c for c in _MATRIX_COLS if c.split('_')[0] in _VARIANT_METRICS]
_SEED_MEAN_COLS = [c for c in _MATRIX_COLS if c not in ('source', 'model', 'pass')]
_SEED_COLS = (['source', 'model', 'pass', 'n_seeds', 'seeds']
              + [c for m in _SEED_MEAN_COLS
                 for c in ((m, f'{m}_sd') if m in _SEED_METRIC_COLS else (m,))])


def aggregate_seeds(rows: list[dict]) -> list[dict]:
    """
    Collapse rows whose source differs only in its _seed<N> token into one row
    of means, with a population sd next to every metric (0 for a lone seed).

    Only the few-shot passes really vary with the seed — it picks the examples.
    A ZS row aggregated the same way is a free control: its sd should be 0,
    since greedy decoding on identical input cannot depend on the seed.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_seed_group(r['source']), []).append(r)

    out = []
    for sid, grp in groups.items():
        agg = {'source': sid, 'model': grp[0]['model'], 'pass': grp[0]['pass'],
               'n_seeds': len(grp),
               'seeds': '+'.join(sorted(m.group(1) for r in grp
                                        if (m := _SEED_RE.search(r['source']))))}
        for col in _SEED_MEAN_COLS:
            vals = [r[col] for r in grp if isinstance(r[col], (int, float))]
            if not vals:
                agg[col] = ''
                continue
            mean = float(np.mean(vals))
            agg[col] = (int(round(mean)) if col in ('n', 'total_chars')
                        else round(mean, 6))
            if col in _SEED_METRIC_COLS:
                agg[f'{col}_sd'] = round(float(np.std(vals)), 6)
        out.append(agg)
    return out


# ── Console rendering ─────────────────────────────────────────────────────────

def _num(value, spec: str) -> str:
    """Format a number, or '-' if the run did not record it."""
    return format(value, spec) if isinstance(value, (int, float)) else '-'


def print_matrix(rows: list[dict]) -> None:
    """
    Readable console rendering of the wide table. Rows carrying "<metric>_sd"
    (i.e. seed aggregates) print as mean ±sd.
    """
    if not rows:
        return
    name_w = max(len(r['source']) for r in rows) + 2
    with_sd = 'cer_sd' in rows[0]
    w = 17 if with_sd else 10

    def cell(row: dict, key: str, spec: str) -> str:
        text = format(row[key], spec)
        if f'{key}_sd' in row:
            text += f" ±{row[f'{key}_sd']:{spec}}"
        return f'{text:>{w}}'

    def block(title: str, columns: tuple[str, ...], render) -> None:
        header = f"{title:<{name_w}}{'n':>6}" + ''.join(f'{c:>{w}}' for c in columns)
        print(header)
        print('-' * len(header))
        for r in rows:
            print(f"{r['source']:<{name_w}}{r['n']:>6}" + render(r))
        print()

    for vname in VARIANTS:
        s = _suffix(vname)
        rows.sort(key=lambda r: r[f'cer{s}'])
        block(vname, _VARIANT_METRICS,
              lambda r, s=s: cell(r, f'cer{s}', '.4f') + cell(r, f'wer{s}', '.4f')
                             + cell(r, f'acc{s}', '.1%'))

    # Run cost does not depend on the normalisation variant.
    rows.sort(key=lambda r: r['cer'])
    block('run cost', ('VRAM', 'items/s'),
          lambda r: f"{_num(r['vram_gib'], '.1f'):>{w}}"
                    + f"{_num(r['items_per_s'], '.2f'):>{w}}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HTR predictions against annotations",
        epilog="Every (file × pass) is scored as its own entity, so ZS and "
               "5_Shot from one run appear as two rows.")
    parser.add_argument('--dataset', default=None, metavar='NAME',
                        help='Dataset name. Selects every JSON in --prediction-dir whose '
                             'filename contains it, and names the output CSV.')
    parser.add_argument('--prediction-dir', dest='prediction_dir', default=None, metavar='DIR',
                        help='Folder of prediction JSONs (filtered by --dataset)')
    parser.add_argument('--predictions', nargs='+', default=None, metavar='JSON',
                        help='Explicit prediction JSONs, instead of --prediction-dir')
    parser.add_argument('--annotations', default=None,
                        help='Annotations JSON with the reference transcriptions. Only needed '
                             'for datasets whose prediction JSONs carry no "ground_truth" per '
                             'item.')
    parser.add_argument('--pass', '--passes', dest='pass_labels', nargs='+', default=None,
                        metavar='LABEL',
                        help='Evaluate only these pass keys (e.g. --pass ZS 5_Shot lora_zs). '
                             'Omit to evaluate every pass found in each file. Files lacking a '
                             'requested pass are skipped with a warning.')
    parser.add_argument('--output', default=None,
                        help='Output CSV (default: <dataset>_eval.csv, else <stem>_eval.csv '
                             'for a single file / htr_eval_table.csv for several)')
    parser.add_argument('--per-image', default=None, metavar='CSV',
                        help='Also write one row per image per source (for significance tests)')
    args = parser.parse_args()

    if bool(args.prediction_dir) == bool(args.predictions):
        parser.error('give either --prediction-dir (with --dataset) or --predictions')

    if args.prediction_dir:
        pred_dir = Path(args.prediction_dir)
        if not pred_dir.is_dir():
            parser.error(f"prediction dir not found: {pred_dir}")
        pred_paths = discover_prediction_files(pred_dir, args.dataset)
        if not pred_paths:
            parser.error(f"no JSON in {pred_dir} matching '{args.dataset}'")
        print(f"Prediction files ({len(pred_paths)}) in {pred_dir}"
              + (f" matching '{args.dataset}'" if args.dataset else ''))
        for p in pred_paths:
            print(f"  {p.name}")
    else:
        pred_paths = [Path(p) for p in args.predictions]
        missing = [str(p) for p in pred_paths if not p.exists()]
        if missing:
            parser.error(f"predictions file(s) not found: {', '.join(missing)}")

    # Optional: a separate annotations file. Prediction JSONs that carry their
    # own "ground_truth" need none, so this stays empty for benchmark datasets.
    annotations: dict = {}
    if args.annotations:
        ann_path = Path(args.annotations)
        if not ann_path.exists():
            parser.error(f"annotations file not found: {ann_path}")
        print(f"Loading annotations:  {ann_path}")
        annotations = load_annotations(ann_path)
        print(f"  {len(annotations)} annotated images")

    evaluated: list[EvaluatedSource] = []
    per_image_rows: list[dict] = []
    checked_overlap = False

    for pred_path in pred_paths:
        all_passes, model_name, embedded_gt, remarks = load_predictions(pred_path)
        available = list(all_passes)

        refs = annotations or annotations_from_ground_truth(embedded_gt)
        if not refs:
            print(f"  [skip] {pred_path.name}: no ground truth in the file and no "
                  f"--annotations given", file=sys.stderr)
            continue

        if not checked_overlap and all_passes:
            first_pass = next(iter(all_passes.values()))
            overlap = set(refs) & set(first_pass)
            if not overlap:
                print("  WARNING: zero key overlap between references and predictions!",
                      file=sys.stderr)
                print(f"  reference  key example: {next(iter(refs))}", file=sys.stderr)
                print(f"  prediction key example: {next(iter(first_pass))}", file=sys.stderr)
            else:
                print(f"  {len(overlap)} keys matched between references and predictions")
            checked_overlap = True

        if args.pass_labels:
            wanted = [p for p in args.pass_labels if p in all_passes]
            if not wanted:
                print(f"  [skip] {pred_path.name}: none of {args.pass_labels} present "
                      f"(has {available})", file=sys.stderr)
                continue
        else:
            wanted = available

        for label in wanted:
            results, n_skip = evaluate(refs, all_passes[label])
            sid = source_id(pred_path, label)
            evaluated.append(EvaluatedSource(sid, model_name,
                                             pass_display(pred_path, label), results,
                                             peak_vram_gib(remarks, label),
                                             items_per_second(remarks, label)))
            print(f"  {sid:<55} evaluated {len(results):>5}  skipped {n_skip}")
            if args.per_image:
                for r in results:
                    per_image_rows.append({
                        'source': sid, 'filename': r['filename'],
                        'complicated': r['complicated'],
                        'ref_chars': r['variants']['base']['ref_chars'],
                        **{f'{m}{_suffix(v)}': r['variants'][v][m]
                           for v in VARIANTS for m in ('cer', 'wer', 'correct')},
                    })

    if not evaluated:
        print("Nothing evaluated — check --pass and the prediction files.", file=sys.stderr)
        sys.exit(1)

    rows = build_matrix_rows(evaluated)
    if args.output:
        out_path = Path(args.output)
    elif args.dataset:
        out_path = Path(f'{args.dataset}_eval.csv')
    elif len(pred_paths) == 1:
        out_path = Path(_strip_pred_suffix(pred_paths[0]) + '_eval.csv')
    else:
        out_path = Path('htr_eval_table.csv')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_MATRIX_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport written to: {out_path}\n")
    print_matrix(rows)

    # Runs that differ only in their seed are averaged into a second table.
    seed_rows = aggregate_seeds(rows)
    if any(r['n_seeds'] > 1 for r in seed_rows):
        seed_path = out_path.with_name(out_path.stem + '_seedmean.csv')
        with open(seed_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_SEED_COLS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(seed_rows)
        print(f"Seed averages written to: {seed_path}  (mean ± sd)\n")
        print_matrix(seed_rows)

    if args.per_image and per_image_rows:
        with open(args.per_image, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0]))
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"Per-image rows written to: {args.per_image}  ({len(per_image_rows)} rows)")


if __name__ == '__main__':
    main()
