# Data Unification Phase

## Scope
This document describes the unified-build phase of the data pipeline.

This phase reads raw datasets and builds one unified JSONL corpus for downstream augmentation and training.

Source datasets:
1. Twitter15
2. Twitter16
3. RumourEval (train and test merged)
4. PHEME

Primary output:
1. Datasets/unified_records.jsonl

## Directory Responsibilities
1. src/data_prep/: core unification library code (loaders, normalization, schema conversion, inspection)
2. scripts/preprocess.py: top-level orchestration entry point
3. scripts/inspect_datasets.py: inspection-only entry point

## Read-Only Inputs
Raw data is read from .datasets_raw/ and not modified.

Expected layout:
1. .datasets_raw/twitter15/
2. .datasets_raw/twitter16/
3. .datasets_raw/RumourEval/train/ and .datasets_raw/RumourEval/test/ (or)
4. .datasets_raw/RumourEval/rumoureval-2019-training-data/ and .datasets_raw/RumourEval/rumoureval-2019-test-data/
5. .datasets_raw/PHEME/all-rnr-annotated-threads/

## Phase Stages
1. Structure inspection
- Validates source folders and expected file presence.
- Writes Datasets/inspection.json.

2. Dataset loading
- Twitter15/16: label.txt + source_tweets.txt
- RumourEval: recursively loads JSON samples from train and test folders and unifies them.
  - Source posts only are ingested (raw.json and source-tweet/source-tweets files); replies are excluded.
- PHEME: loads threads from all-rnr-annotated-threads.

3. Label normalization
- Normalized label space:
  - false
  - true
  - unverified
  - non-rumor

4. Unified schema conversion
- Converts samples to UnifiedRecord shape.

5. Optional enrichment hook
- Emotion annotation (`--enable-emotion`) during unified build.

Note:
Fact verification is not part of this phase; it is handled in the final model input phase.

6. Diversity shuffle
- Unified records are shuffled with a deterministic seed before write.

7. Artifact write
- Writes unified_records.jsonl.
- Writes stats.json and inspection.json.

## Output Artifacts
Generated under Datasets/:
1. unified_records.jsonl
2. stats.json
3. inspection.json

## Example Commands
Run inspection only:

```powershell
python scripts/inspect_datasets.py --data-root .datasets_raw --output Datasets/inspection.json
```

Run unified-build phase only:

```powershell
python scripts/preprocess.py --enable-unified-build --data-root .datasets_raw --output-dir Datasets
```

Run unified-build with emotion annotation:

```powershell
python scripts/preprocess.py --enable-unified-build --enable-emotion
```
