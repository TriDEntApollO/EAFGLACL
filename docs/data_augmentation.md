# Data Augmentation Pipeline

## Purpose
This document describes the Gemini-based augmentation pipeline implemented in:
1. `scripts/preprocess.py`
2. `src/augmentation/pipeline.py`
3. `src/augmentation/gemini_augmenter.py`

The pipeline reads unified source records from JSONL, generates rewritten candidates, and appends accepted rows to an output JSONL file in real time.

## Inputs and Outputs

### Input
1. Default input file: `Datasets/unified_records.jsonl` when `--output-dir Datasets` is used.
2. Override input with `--augmentation-input <path>`.

### Outputs
1. Augmented rows JSONL:
   - Default: `<output-dir>/augmented_records.jsonl`
   - Override: `--augmentation-output <path>`
2. Augmentation stats JSON:
   - Default: `<output-dir>/augmentation_stats.json`
   - Override: `--augmentation-stats-output <path>`

## Run Commands

Run augmentation only (recommended):

```powershell
# inside the virtual env
python ./scripts/preprocess.py --enable-augmentation --output-dir Datasets
```

Run a small smoke test:

```powershell
# inside the virtual env
python ./scripts/preprocess.py --enable-augmentation --output-dir Datasets --augmentation-max-records 50
```

## Environment Configuration

Required in `.env`:
1. `GEMINI_API_KEY`
2. `PROJECT_NAME` (format: `projects/<number>`)
3. `PROJECT_NUMBER`
4. `GEMINI_MODEL`

Optional rate and sizing controls:
1. `AUGMENTATION_TARGET_ENTRIES`
2. `RPM`
3. `TPM`
4. `RPD`

CLI values override `.env` values where provided.

## CLI Options

Common augmentation flags:
1. `--enable-augmentation`
2. `--augmentation-input`
3. `--augmentation-output`
4. `--augmentation-stats-output`
5. `--augmentation-max-records`
6. `--augmentation-candidates-per-record`
7. `--gemini-model`
8. `--augmentation-temperature`
9. `--augmentation-top-p`
10. `--augmentation-max-output-tokens`
11. `--augmentation-target-entries`
12. `--augmentation-rpm-limit`
13. `--augmentation-tpm-limit`
14. `--augmentation-rolling-window-seconds`

## Execution Flow

1. Load `.env` and CLI config.
2. Validate preflight:
   - API key present.
   - Project fields present and consistent.
   - Model exists for the current key/project.
   - Lightweight generation call succeeds.
3. Resolve effective limits:
   - Start from CLI and `.env` values.
   - Fill missing values from API-discovered metadata/quota details when available.
4. Process source records in batches.
5. Generate candidates per batch.
6. Parse structured response and map candidates to source `record_id`.
7. Append each accepted candidate row immediately to output JSONL.
8. Log per-batch progress and limiter state to terminal.
9. Write summary stats JSON.

## Output Row Schema

Each emitted JSONL row currently contains:
1. `source_record_id`
2. `dataset`
3. `label`
4. `source_text`
5. `candidate_text`
6. `generation_model`
7. `prompt_version`
8. `created_at_utc`

## Stats Schema (Key Fields)

`augmentation_stats.json` includes:
1. `input_records`
2. `generated_candidates`
3. `accepted_pairs`
4. `failed_generations`
5. `acceptance_rate`
6. `model_name`
7. `requested_batches`
8. `rpm_limit`
9. `tpm_limit`
10. `rpd_limit`
11. `rate_limit_wait_seconds`
12. `input_path`
13. `output_path`

## Rate Limit and Retry Behavior

1. Rolling-window gating is applied for RPM and TPM when limits are available.
2. Batch size is reduced adaptively if estimated tokens exceed TPM.
3. Retries use API-provided retry delay when present:
   - `retryDelay: "Ns"`
   - `Please retry in Xs`
4. If API retry delay is absent, exponential backoff is used.

## Terminal Logging

Per run and per batch logs include:
1. Preflight start and pass/failure.
2. Effective limits in use.
3. Retry delay on rate-limit events.
4. Batch number and dataset progress.
5. Generated and appended counts.
6. Current request/token window usage.

## Troubleshooting

### `429 RESOURCE_EXHAUSTED`
Cause:
1. Per-model free-tier request/token quota reached.

What to do:
1. Wait for quota reset and retry.
2. Use a model with available quota for your project.
3. Reduce batch size and candidate count for smoke runs.

### Preflight model not available
Cause:
1. Model name in `.env` is not available to your key/project.

What to do:
1. Set `GEMINI_MODEL` to an available model.
2. Or pass `--gemini-model <model>`.

### Empty/low output volume
Cause:
1. Quota exhaustion mid-run.
2. Batch response parsing mismatch for a specific model response shape.

What to do:
1. Run with `--augmentation-max-records 1` and inspect logs first.
2. Confirm rows are appended in real time to output JSONL.
