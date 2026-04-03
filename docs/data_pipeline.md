# EAFGLACL Data Pipeline

## Scope
This document describes the end-to-end preprocessing pipeline used to unify four misinformation datasets into one JSONL corpus for model training:

1. Twitter15
2. Twitter16
3. Weibo v2
4. PHEME

The primary output is a unified dataset written directly under `Datasets/`.

## Read-Only Constraint
Raw dataset folders are treated as read-only inputs under `.datasets_raw/`:

1. `.datasets_raw/twitter15/`
2. `.datasets_raw/twitter16/`
3. `.datasets_raw/Weibo/`
4. `.datasets_raw/PHEME/`

All generated files are written only to `Datasets/`.

## Entry Scripts
Pipeline execution is done via:

1. `scripts/inspect_datasets.py`
2. `scripts/preprocess.py`

## Commands Used
From repository root:

```powershell
& "f:/Important/Programming Projects/LALC/.venv/Scripts/python.exe" "scripts/inspect_datasets.py" --data-root ".datasets_raw" --output "Datasets/inspection.json"
& "f:/Important/Programming Projects/LALC/.venv/Scripts/python.exe" "scripts/preprocess.py" --data-root ".datasets_raw" --output-dir "Datasets"
```

## Pipeline Stages
1. Structure inspection
- Confirms required dataset layout and expected source files/directories.
- Produces `inspection.json`.

2. Dataset-specific loading
- Twitter15/Twitter16: load from `label.txt` and `source_tweets.txt`.
- Weibo: load `fake_news/` and `real_news/` directly under `.datasets_raw/Weibo/`.
- PHEME: load events from `all-rnr-annotated-threads/`.

3. Label normalization
- Unifies labels into one shared space:
  - `false`
  - `true`
  - `unverified`
  - `non-rumor`

4. Unified schema conversion
- Every sample is converted into the shared record format (`UnifiedRecord`) with mandatory and optional fields.

5. Stratified split generation
- Creates `train/val/test` splits stratified by label.
- Default ratios: 0.8/0.1/0.1.

6. Artifact writing
- Writes full corpus + splits in JSONL.
- Writes aggregate dataset statistics in JSON.

## Unified Schema
Each output row includes:

1. `record_id` (string)
2. `dataset` (string)
3. `label` (string)
4. `text` (string)
5. Optional fields:
   - `language`
   - `event`
   - `thread_id`
   - `source_id`
   - `emotion`
   - `emotion_confidence`
   - `fact_consistency_score`
   - `comments` (list of strings)
   - `metadata` (dict)

## Output Artifacts
Generated under `Datasets/`:

1. `inspection.json`: layout/availability checks
2. `unified_records.jsonl`: full merged dataset
3. `train.jsonl`: training split
4. `val.jsonl`: validation split
5. `test.jsonl`: test split
6. `stats.json`: totals, distributions, split counts

## Current Run Summary
From `Datasets/stats.json`:

1. `total_records`: 20062
2. `by_dataset`:
   - twitter15: 1490
   - twitter16: 818
   - weibo: 11329
   - pheme: 6425
3. `by_label`:
   - false: 6874
   - true: 7314
   - non-rumor: 4601
   - unverified: 1273
4. `split_sizes`:
   - train: 16048
   - val: 2005
   - test: 2009

## Validation Checklist
1. `inspection.json` indicates all four datasets are present.
2. All six output files exist in `Datasets/`.
3. `train + val + test == total_records`.
4. Labels are in the normalized label space.

## Next Integration Step
Use `Datasets/train.jsonl` as input to LLM augmentation and then apply:

1. Emotion consistency filtering
2. External fact verification scoring

before final contrastive training.