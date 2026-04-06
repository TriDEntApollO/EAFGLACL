# Runtime Fact Verification in Final Evaluation

This document explains how fact verification works at model runtime in the evaluation stage and how the fact consistency score is used for final prediction.

## Where it runs

Runtime fact verification is applied in:

- `scripts/final_evaluation.py`

The workflow is:

1. Load dataset rows (`--data-path`, default `Datasets/complete_data.jsonl`).
2. Run live CL inference from exported model weights (`--model-weights`, default `outputs/cl_outputs/cl_model_weights.pt`).
3. Obtain baseline predicted label (`predicted_label`) for each row.
4. If fact verification is enabled, compute `fact_score` for each row.
5. Convert `fact_score` to a fact label (`fact_label`).
6. Compute `final_label` used by final metrics.

## Providers

Use `--fact-provider` to select runtime fact verification flow:

- `gemini_plus_wikipedia` (default): Gemini claim flow + Wikipedia evidence + Gemini confidence.
- `wikipedia_only`: Wikipedia-based consistency score only.

Disable fact verification entirely:

- `--disable-fact-verification`

## Fact consistency score to label mapping

For each evaluated text, a scalar `fact_score` in `[0, 1]` is mapped as:

- `fact_score >= 0.66` -> `fact_label = "true"`
- `fact_score <= 0.34` -> `fact_label = "false"`
- otherwise -> `fact_label = "unverified"`

## Final prediction rule

Final label logic is:

- If baseline `predicted_label == "non-rumor"`, keep `final_label = "non-rumor"`.
- Otherwise, use fact verification output: `final_label = fact_label`.

This means fact consistency score only re-assigns rumor-class predictions (`false`, `true`, `unverified`), while non-rumor predictions are preserved.

## Output fields

Per-record output (`--output-path`, default `outputs/final_outputs/final_evaluation_records.json`) includes:

- `predicted_label`: CL baseline label.
- `fact_score`: runtime score (or `null` when disabled).
- `fact_label`: mapped label from fact score.
- `final_label`: label used for final pipeline evaluation.
- `fact_verification_applied`: boolean flag.

Aggregate metrics (`--metrics-path`, default `outputs/final_outputs/final_evaluation_metrics.json`) include:

- `accuracy_without_fact_verification`
- `accuracy_with_current_pipeline`
- `rumor_only_accuracy_without_fact_verification`
- `rumor_only_accuracy_with_current_pipeline`
- `accuracy_delta`
- `fact_verification_enabled`
- `fact_verification_flow`

## Example commands

Run with default runtime fact verification:

```bash
python scripts/final_evaluation.py
```

Run with Wikipedia-only fact verification:

```bash
python scripts/final_evaluation.py --fact-provider wikipedia_only
```

Run baseline-only (fact verification disabled):

```bash
python scripts/final_evaluation.py --disable-fact-verification
```

## Notes

- If legacy TensorFlow checkpoint shards (`.index`/`.data`) are found and `.pt` is missing, evaluation uses compatibility handoff input to keep the pipeline runnable.
- For best reproducibility, run with the project virtual environment Python.
