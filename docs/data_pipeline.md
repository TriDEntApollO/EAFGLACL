# EAFGLACL Data Pipeline

## Scope
This document describes the current preprocessing pipeline used to build one unified JSONL corpus for downstream augmentation and training.

Source datasets:
1. Twitter15
2. Twitter16
3. RumourEval (train and test merged)
4. PHEME

Primary output:
1. Datasets/unified_records.jsonl

## Directory Responsibilities
1. src/dataa_prep/: core preprocessing library code (loaders, normalization, enrichment, unified builder, inspection)
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

## Pipeline Stages
1. Structure inspection
- Validates source folders and expected file presence.
- Writes Datasets/inspection.json.

2. Dataset loading
- Twitter15/16: label.txt + source_tweets.txt
- RumourEval: recursively loads JSON samples from train and test folders and unifies them
  - Source posts only are ingested (raw.json and source-tweet/source-tweets files); replies are excluded.
- PHEME: loads threads from all-rnr-annotated-threads

3. Label normalization
- Normalized label space:
  - false
  - true
  - unverified
  - non-rumor

4. Unified schema conversion
- Converts samples to UnifiedRecord shape.

5. Optional enrichment hooks
- Emotion annotation (enable_emotion)
- Fact consistency scoring (enable_fact)

6. Diversity shuffle
- Unified records are shuffled with a deterministic seed before write.

7. Artifact write
- Writes only unified_records.jsonl (split generation deferred to a later stage).
- Writes stats.json and inspection.json.

## Output Artifacts
Generated under Datasets/:
1. unified_records.jsonl
2. stats.json
3. inspection.json

When augmentation is enabled:
1. augmented_pairs.jsonl
2. augmentation_stats.json

## Augmentation Input and Rate Control
Augmentation reads from unified_records.jsonl.

Batch behavior:
1. Targets 200 source entries per Gemini request.
2. Estimates token usage per request and reduces batch size when needed.
3. Enforces TPM limit (default 250000).
4. Waits for minute window reset before sending the next request when budget would be exceeded.

Supported controls:
1. CLI: --augmentation-target-entries, --augmentation-tpm-limit
2. .env: AUGMENTATION_TARGET_ENTRIES, GEMINI_TPM_LIMIT

## Example Commands
Run inspection only:

```powershell
python scripts/inspect_datasets.py --data-root .datasets_raw --output Datasets/inspection.json
```

Run preprocessing (unified dataset only):

```powershell
python scripts/preprocess.py --data-root .datasets_raw --output-dir Datasets
```

Run augmentation from unified records with rate control:

```powershell
python scripts/preprocess.py --enable-augmentation --augment-only --augmentation-target-entries 200 --augmentation-tpm-limit 250000
```
