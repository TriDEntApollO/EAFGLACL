# Model Training Pipeline

This document describes the training stage of the project pipeline.

Pipeline context:

Data preprocessing -> Data augmentation -> Emotion filtering -> Contrastive learning + classification training

## Purpose

The training pipeline consumes the merged dataset from the emotion-filtering stage and trains:
1. A contrastive encoder using InfoNCE loss.
2. A classifier head for rumor label prediction.

## Input and output

### Input

Default input file:
- `Datasets/complete_data.jsonl`

Override input with:
- `--data-path <path>`

### Output

Default output directory:
- `outputs/cl_outputs/`

Contains:
- `cl_outputs_for_fact_verification.json`
- `cl_model_weights.*`
- `classification_report.txt`
- `training_history.json`

## Run commands

From repository root:

```powershell
python scripts/training.py
```

Train on dedicated GPU (fails fast if no dedicated GPU is detected):

```powershell
python scripts/training.py --use-gpu
```

Override data and output paths:

```powershell
python scripts/training.py --data-path Datasets/complete_data.jsonl --output-dir outputs/cl_outputs
```

Optional threshold override for CL pair quality:

```powershell
python scripts/training.py --emotion-sim-threshold 0.9
```

## Training phases

1. Contrastive pre-training
- Uses BERT encoder + projection head.
- Optimizes InfoNCE objective on emotion-consistent pairs.

2. Supervised classifier training
- Uses frozen encoder representations.
- Trains a softmax classifier over the unified label space.

3. Handoff artifact generation
- Exports prediction probabilities and dense embeddings for downstream fact-verification workflows.

## Configuration reference

Primary runtime entrypoint:
- `scripts/training.py`

Core model implementation:
- `src/models/cl_model.py`

Important configurable parameters include:
- Data path and output directory.
- Contrastive hyperparameters (temperature, epochs, batch size).
- Classifier epochs.
- Emotion similarity threshold.
- Device mode via `--use-gpu` (dedicated GPU required).

## Unified label space

- `false`
- `true`
- `unverified`
- `non-rumor`
