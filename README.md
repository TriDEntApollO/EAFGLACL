# EAFGLACL: Emotion-Aware, Fact-Grounded LLM-Augmented Contrastive Learning for Misinformation Detection in Social Networks

This repository implements an extended LACL-style pipeline for misinformation detection across:

- `Twitter15`
- `Twitter16`
- `Weibo v2`
- `PHEME`

The current focus is data pipeline and preprocessing, with modular hooks for:

- Emotion-preserving LLM augmentation
- External fact verification
- Contrastive + classification training

## Dataset sources

- Twitter15/Twitter16: https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017
- Weibo v2: https://www.kaggle.com/datasets/yangzhou32/socialnet-weibo-version-2
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
  Weibo/
    fake_news/                          # moved directly under Weibo/
    real_news/                          # moved directly under Weibo/
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run preprocessing:

```bash
python scripts/preprocess.py
```

Enable emotion + fact-aware enrichment:

```bash
python scripts/preprocess.py --enable-emotion --enable-fact
```

Fast smoke run with a per-dataset cap:

```bash
python scripts/preprocess.py --max-per-dataset 200
```

Inspect dataset structure only (fast, no full scan of all records):

```bash
python scripts/inspect_datasets.py
```

## Output artifacts

After preprocessing, the output directory contains:

- `unified_records.jsonl` (all unified examples)
- `train.jsonl`, `val.jsonl`, `test.jsonl` (stratified splits)
- `stats.json` (label + dataset stats)
- `inspection.json` (dataset structure checks)

By default these files are written directly under `Datasets/`.

## Unified label space

- `false`
- `true`
- `unverified`
- `non-rumor`

## Project structure

```text
src/
  data/                  # dataset loaders + normalization + split builder
  augmentation/          # LLM augmentation + emotion-preserving filtering hooks
  fact/                  # external verification hooks (Wikipedia-backed)
  models/                # baseline encoder/training interfaces (scaffold)
  preprocessing/         # preprocessing schemas/loaders scaffolds
scripts/
  preprocess.py
  inspect_datasets.py
```
