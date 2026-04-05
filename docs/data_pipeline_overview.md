# Data Pipeline Overview

## Scope
This document describes the end-to-end data pipeline phases executed by scripts/preprocess.py.

## Phase Sequence
1. Unified build (`--enable-unified-build`)
- Builds Datasets/unified_records.jsonl from raw datasets.
- Optional: add emotion metadata with `--enable-emotion`.

2. Augmentation (`--enable-augmentation`)
- Generates LLM-based candidates from unified records.
- Writes Datasets/augmented_records.jsonl and Datasets/augmentation_stats.json.

3. Emotion filtering (`--enable-emotion-filter`)
- Filters augmentation candidates by emotion consistency.
- Writes Datasets/filtered_augmented_data.jsonl.
- Merges with unified data into Datasets/complete_data.jsonl.

## Phase Selection Behavior
1. Each phase runs only when its flag is passed.
2. Phases can be run independently or in one command.
3. If no phase flag is passed, preprocess.py prints a short phase help menu and exits.

## Common Run Patterns
Unified build only:

```powershell
python scripts/preprocess.py --enable-unified-build
```

Augmentation only (requires existing unified_records.jsonl):

```powershell
python scripts/preprocess.py --enable-augmentation
```

Emotion filtering only (requires existing unified_records.jsonl and augmented_records.jsonl):

```powershell
python scripts/preprocess.py --enable-emotion-filter
```

Full data pipeline:

```powershell
python scripts/preprocess.py --enable-unified-build --enable-augmentation --enable-emotion-filter
```
