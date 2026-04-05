# EAFGLACL Development Roadmap

## Objective
Build an emotion-aware and fact-grounded misinformation detector that extends LACL while prioritizing robust preprocessing and data quality.

## Research Gaps Covered
1. Emotional consistency during augmentation
- Problem: LLM augmentation can change emotional tone.
- Mitigation: Use an emotion classifier to retain only emotion-consistent candidates.

2. No external factual verification
- Problem: Reliance on model-internal world knowledge introduces hallucination risk.
- Mitigation: Add claim-level external verification and a fact-consistency score.

## Phase Plan
1. Data pipeline and preprocessing (current priority)
- Verify folder layout for Twitter15, Twitter16, Weibo, and PHEME.
- Normalize labels into one space: false, true, unverified, non-rumor.
- Build unified schema with text, comments, thread metadata, and optional enrichment fields.
- Generate train/val/test splits and dataset statistics.

2. LLM augmentation with emotional consistency
- Generate candidate positive pairs using LLM prompts.
- Filter candidates by emotion consistency against source texts.
- Track acceptance rate and class balance effects.

3. Fact-grounded feature enrichment
- Extract claim-like spans.
- Verify claims against trusted sources (Wikipedia first, extensible to other APIs).
- Produce record-level and claim-level fact consistency features.

4. Contrastive model training
- Baseline transformer encoder + supervised head.
- Contrastive objective with label-aware and augmentation-aware positive pairs.
- Ablation runs: baseline vs emotion-only vs fact-only vs combined.

5. Evaluation and optimization
- Per-dataset and cross-dataset metrics.
- Error slices: sarcasm, emotionally charged rumors, fact-conflicting cases.
- Throughput and memory optimization for large-scale training.

## Suggested Team Role Split (6 members)
1. Data engineering lead: ingestion, normalization, quality checks.
2. Augmentation lead: LLM prompts, pair generation, filtering.
3. Emotion modeling lead: classifier selection, calibration, QA.
4. Fact verification lead: claim extraction, source matching, scoring.
5. Modeling lead: CL objective, encoder training, evaluation loops.
6. Integration and MLOps lead: scripts, experiments, reproducibility, docs.

## Immediate Next Deliverables
1. Completed: run full preprocessing on Twitter15, Twitter16, Weibo, and PHEME.
2. Completed: generate unified JSONL artifacts under `Datasets/`.
3. Completed: review dataset and label distributions from `Datasets/stats.json`.
4. Completed: publish unification and pipeline docs in `docs/data_unification.md` and `docs/data_pipeline_overview.md`.
5. Next: freeze v1 data artifacts and begin augmentation plus enrichment experiments.
