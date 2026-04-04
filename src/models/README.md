# LACL — Contrastive Learning Module

## Overview

This module is **Stage 4** of the full LACL pipeline:

```
Data Preprocessing → Data Augmentation → Emotion Filtering → [Final Dataset]
→ Contrastive Learning (THIS MODULE) → Model
→ Fact Verification → Final Model
```

It takes the `Complete_data.jsonl` from the emotion filtering stage and:
1. **Phase 1** — Contrastive pre-training using InfoNCE loss (BERT + MLP projector)
2. **Phase 2** — Classifier training on frozen encoder (4-class label prediction)
3. **Exports** embeddings + predictions for the Fact Verification team

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
# Place Complete_data.jsonl in the same folder, then:
python cl_model.py
```

---

## Config (cl_model.py → class Config)

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `Complete_data.jsonl` | Input dataset path |
| `OUTPUT_DIR` | `cl_outputs/` | Where all outputs are saved |
| `BERT_MODEL_NAME` | `bert-base-uncased` | Encoder backbone |
| `MAX_SEQ_LEN` | 128 | Token max length |
| `PROJECTION_DIM` | 128 | MLP output dimension |
| `TEMPERATURE` | 0.07 | τ in InfoNCE loss |
| `CL_EPOCHS` | 5 | Contrastive pre-training epochs |
| `CL_BATCH_SIZE` | 32 | Pairs per CL batch |
| `CLF_EPOCHS` | 10 | Classifier training epochs |
| `EMOTION_SIM_THRESHOLD` | 0.85 | Min emotion similarity for valid pair |

---

## Outputs (in `cl_outputs/`)

| File | Description |
|---|---|
| `cl_outputs_for_fact_verification.json` | **Handoff to Fact Verification** — contains text, predicted label, label probabilities, and 768-dim BERT embedding per sample |
| `cl_model_weights.*` | Saved TF model weights |
| `classification_report.txt` | Per-class precision, recall, F1, accuracy |
| `training_history.json` | CL loss per epoch + classifier val accuracy |

---

## Handoff Format (for Fact Verification team)

Each record in `cl_outputs_for_fact_verification.json`:

```json
{
  "text": "the original tweet text",
  "predicted_label": "false",
  "true_label": "false",
  "label_probs": {
    "false": 0.81,
    "non-rumor": 0.10,
    "true": 0.05,
    "unverified": 0.04
  },
  "embedding": [0.23, -0.11, ...]   // 768-dim BERT [CLS] vector
}
```

---

## Architecture

```
Input Text
    │
    ▼
BERT-base-uncased (shared encoder)
    │
    ▼  [CLS] embedding (768-dim)
    │
    ├──── [CL Phase] ────► MLP Projector (768→256→128) ──► InfoNCE Loss
    │
    └──── [Classifier Phase, encoder frozen] ────► Dense + Softmax ──► Label
```

---

## Labels

| Label | Description |
|---|---|
| `non-rumor` | Verified accurate content |
| `false` | Confirmed false rumor |
| `true` | Confirmed true rumor |
| `unverified` | Rumor, verification status unknown |
