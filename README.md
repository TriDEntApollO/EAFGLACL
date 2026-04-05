# EAFGLACL: Emotion-Aware, Fact-Grounded LLM-Augmented Contrastive Learning for Misinformation Detection in Social Networks

This repository implements an extended LACL-style pipeline for misinformation detection across:

- `Twitter15`
- `Twitter16`
- `RumourEval`
- `PHEME`

The current focus is data pipeline and preprocessing, with modular hooks for:

- Emotion-preserving LLM augmentation
- Emotion-consistency filtering for augmented pairs
- External fact verification (post-training/final model input phase)
- Contrastive + classification training

## Pipeline phases

The current end-to-end workflow is organized into the following phases:

1. Preprocessing
- Builds `Datasets/unified_records.jsonl` plus inspection/stat artifacts.

2. Augmentation
- Generates `Datasets/augmented_records.jsonl` using Gemini.

3. Emotion filtering
- Filters augmentation quality and writes:
  - `Datasets/filtered_augmented_data.jsonl`
  - `Datasets/complete_data.jsonl`

4. Model training
- Trains contrastive + classifier stages from `Datasets/complete_data.jsonl`.
- Exports artifacts under `outputs/cl_outputs/`.

## Dataset sources

- Twitter15/Twitter16: https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017
- RumourEval: https://www.kaggle.com/datasets/thedevastator/rumoureval-2019-rumour-verification-dataset
- PHEME: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078

## Data locations

- Raw datasets root: `.datasets_raw/`
- Processed/unified outputs root: `Datasets/`

Raw input folders are treated as read-only during preprocessing.

Expected raw structure:

```text
.datasets_raw/
  PHEME/
    all-rnr-annotated-threads/          # extracted from PHEME_veracity.tar.bz2
  twitter15/                            # extracted from Twitter.zip
  twitter16/                            # extracted from Twitter.zip
  RumourEval/
    train/
    test/
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create your local secrets file:

```bash
copy .env.example .env
```

Then edit `.env` and set `GEMINI_API_KEY`.

Run preprocessing phase only:

```bash
python scripts/preprocess.py --enable-unified-build
```

This now builds only `unified_records.jsonl` (split generation is deferred).

Enable prep-stage emotion enrichment:

```bash
python scripts/preprocess.py --enable-unified-build --enable-emotion
```

Run Gemini augmentation on the train split and write accepted pairs to a separate artifact:

```bash
python scripts/preprocess.py --enable-augmentation
```

Run emotion filtering on existing augmentation outputs:

```bash
python scripts/preprocess.py --enable-emotion-filter
```

Run full data pipeline phases in one command:

```bash
python scripts/preprocess.py --enable-unified-build --enable-augmentation --enable-emotion-filter
```

Run training on filtered + merged data:

```bash
python scripts/training.py
```

Augmentation-only mode on existing unified records:

```bash
python scripts/preprocess.py --enable-augmentation --augmentation-max-records 200
```

Rate-limit aware augmentation controls:

```bash
python scripts/preprocess.py --enable-augmentation --augmentation-target-entries 200 --augmentation-tpm-limit 250000
```

You can also set these in `.env`:

- `AUGMENTATION_TARGET_ENTRIES=200`
- `GEMINI_TPM_LIMIT=250000`

Fast smoke run with a per-dataset cap:

```bash
python scripts/preprocess.py --enable-unified-build --max-per-dataset 200
```

Inspect dataset structure only (fast, no full scan of all records):

```bash
python scripts/inspect_datasets.py
```

## Output artifacts

After preprocessing, the output directory contains:

- `unified_records.jsonl` (all unified examples)
- `stats.json` (label + dataset stats)
- `inspection.json` (dataset structure checks)

By default these files are written directly under `Datasets/`.

When augmentation is enabled, additional files are produced:

- `augmented_records.jsonl` (Gemini-generated augmentation rows)
- `augmentation_stats.json` (generation/acceptance summary)

When emotion filtering is enabled, additional files are produced:

- `filtered_augmented_data.jsonl` (emotion-consistent augmentation rows)
- `complete_data.jsonl` (merged unified + filtered augmentation dataset)

Training outputs are produced under `outputs/cl_outputs/`.

Detailed augmentation documentation:

- `docs/data_augmentation.md`

Pipeline overview documentation:

- `docs/data_pipeline_overview.md`

Unified build phase documentation:

- `docs/data_unification.md`

Detailed emotion filtering documentation:

- `docs/emotion_filtering.md`

Detailed model training documentation:

- `docs/model_training.md`

## Unified label space

- `false`
- `true`
- `unverified`
- `non-rumor`

## Project structure

```text
src/
  data_prep/            # dataset loaders + normalization + unified dataset builder
  augmentation/          # LLM augmentation + emotion-preserving filtering hooks
  fact/                  # external verification hooks (Wikipedia-backed)
  models/                # baseline encoder/training interfaces (scaffold)
scripts/
  preprocess.py
  inspect_datasets.py
  training.py
```
