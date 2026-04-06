"""
=============================================================================
LACL - Contrastive Learning Module (PyTorch)
=============================================================================
Project     : LLM-Augmented Contrastive Learning for Misinformation Detection
Module      : Contrastive Learning (CL) — Stage 4 of pipeline
Input       : Complete_data.jsonl from emotion filtering stage
Output      : Trained model weights + embeddings + predicted labels JSON
              -> handed off to Fact Verification stage

Architecture:
  - Encoder   : bert-base-uncased
  - Projector : 2-layer MLP projection head for contrastive loss
  - Loss      : InfoNCE (NT-Xent)
  - Classifier: Linear head trained on frozen encoder embeddings

Labels      : non-rumor, false, true, unverified (4-class)
Framework   : PyTorch + Transformers
=============================================================================
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


@dataclass
class Config:
    DATA_PATH: str = "Complete_data.jsonl"
    OUTPUT_DIR: str = "cl_outputs"

    BERT_MODEL_NAME: str = "bert-base-uncased"
    MAX_SEQ_LEN: int = 128
    PROJECTION_DIM: int = 128
    HIDDEN_DIM: int = 256

    TEMPERATURE: float = 0.07
    CL_EPOCHS: int = 5
    CL_BATCH_SIZE: int = 32
    CL_LR: float = 2e-5

    CLF_EPOCHS: int = 10
    CLF_BATCH_SIZE: int = 64
    CLF_LR: float = 1e-4
    LOG_EVERY_STEPS: int = 10

    EMOTION_SIM_THRESHOLD: float = 0.85

    VAL_SIZE: float = 0.1
    TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42

    LABEL_NAMES: list[str] = field(
        default_factory=lambda: ["non-rumor", "false", "true", "unverified"]
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_pair_data(config: Config) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    print("\n[1/5] Loading dataset...")

    originals: dict[str, dict] = {}
    augmented: list[dict] = []

    with open(config.DATA_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "source_record_id" in rec:
                augmented.append(rec)
            else:
                originals[rec["record_id"]] = rec

    print(f"  Originals  : {len(originals)}")
    print(f"  Augmented  : {len(augmented)}")

    pairs: list[tuple[str, str, str]] = []
    skipped = 0

    for aug in augmented:
        orig_id = aug.get("source_record_id")
        if orig_id not in originals:
            skipped += 1
            continue

        em_sim = float(aug.get("emotion_similarity", 1.0))
        if em_sim < config.EMOTION_SIM_THRESHOLD:
            skipped += 1
            continue

        orig = originals[orig_id]
        orig_text = str(orig.get("text", "")).strip()
        aug_text = str(aug.get("text", "")).strip()
        label = str(orig.get("label", aug.get("label", "non-rumor")))

        if orig_text and aug_text:
            pairs.append((orig_text, aug_text, label))

    all_samples = [
        (str(rec["text"]).strip(), str(rec["label"]))
        for rec in originals.values()
        if str(rec.get("text", "")).strip() and rec.get("label")
    ]

    print(f"  Valid pairs: {len(pairs)}  (skipped: {skipped})")
    print(f"  All samples for classifier: {len(all_samples)}")

    return pairs, all_samples


class ContrastivePairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, str]], tokenizer, max_len: int) -> None:
        orig_texts = [p[0] for p in pairs]
        aug_texts = [p[1] for p in pairs]

        self.orig_enc = tokenizer(
            orig_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.aug_enc = tokenizer(
            aug_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.orig_enc["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "orig_input_ids": self.orig_enc["input_ids"][idx],
            "orig_attention_mask": self.orig_enc["attention_mask"][idx],
            "aug_input_ids": self.aug_enc["input_ids"][idx],
            "aug_attention_mask": self.aug_enc["attention_mask"][idx],
        }
        if "token_type_ids" in self.orig_enc:
            item["orig_token_type_ids"] = self.orig_enc["token_type_ids"][idx]
        if "token_type_ids" in self.aug_enc:
            item["aug_token_type_ids"] = self.aug_enc["token_type_ids"][idx]
        return item


class ClassifierDataset(Dataset):
    def __init__(self, texts: list[str], label_ids: np.ndarray, tokenizer, max_len: int) -> None:
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }
        if "token_type_ids" in self.enc:
            item["token_type_ids"] = self.enc["token_type_ids"][idx]
        return item


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x


class CLModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, projection_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(self.encoder.config.hidden_size)
        self.projector = ProjectionHead(hidden_size, hidden_dim, projection_dim)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0, :]

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projector(embeddings)

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def infonce_loss(z_orig: torch.Tensor, z_aug: torch.Tensor, temperature: float) -> torch.Tensor:
    z_orig = F.normalize(z_orig, p=2, dim=1)
    z_aug = F.normalize(z_aug, p=2, dim=1)
    logits = torch.matmul(z_orig, z_aug.transpose(0, 1)) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_a + loss_b)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_contrastive(model: CLModel, loader: DataLoader, config: Config, device: torch.device) -> list[float]:
    print("\n[3/5] Contrastive pre-training...")
    model.train()
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.CL_LR)

    losses: list[float] = []

    for epoch in range(1, config.CL_EPOCHS + 1):
        epoch_losses: list[float] = []
        total_steps = len(loader)

        for step, batch in enumerate(loader, start=1):
            batch = _to_device(batch, device)
            optimizer.zero_grad()

            orig_emb = model.encode(
                batch["orig_input_ids"],
                batch["orig_attention_mask"],
                batch.get("orig_token_type_ids"),
            )
            aug_emb = model.encode(
                batch["aug_input_ids"],
                batch["aug_attention_mask"],
                batch.get("aug_token_type_ids"),
            )

            z_orig = model.project(orig_emb)
            z_aug = model.project(aug_emb)
            loss = infonce_loss(z_orig, z_aug, config.TEMPERATURE)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            if step % config.LOG_EVERY_STEPS == 0 or step == total_steps:
                running_avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                print(
                    f"    [CL][Epoch {epoch}/{config.CL_EPOCHS}] "
                    f"step {step}/{total_steps} "
                    f"loss={float(loss.item()):.4f} running_avg={running_avg:.4f}"
                )

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(avg_loss)
        print(f"  Epoch {epoch}/{config.CL_EPOCHS} - CL Loss: {avg_loss:.4f}")

    return losses


def _safe_stratify_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        split = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return np.asarray(split[0]), np.asarray(split[1])
    except ValueError:
        split = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return np.asarray(split[0]), np.asarray(split[1])


def build_classifier_loaders(
    all_samples: list[tuple[str, str]],
    tokenizer,
    label_encoder: LabelEncoder,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, list[str], np.ndarray]:
    texts = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    y = np.asarray(label_encoder.transform(labels), dtype=np.int64)
    indices = np.arange(len(texts))

    idx_train, idx_test = _safe_stratify_split(indices, y, config.TEST_SIZE, config.RANDOM_SEED)
    relative_val = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    idx_train, idx_val = _safe_stratify_split(idx_train, y[idx_train], relative_val, config.RANDOM_SEED)

    train_ds = ClassifierDataset([texts[i] for i in idx_train], y[idx_train], tokenizer, config.MAX_SEQ_LEN)
    val_ds = ClassifierDataset([texts[i] for i in idx_val], y[idx_val], tokenizer, config.MAX_SEQ_LEN)
    test_ds = ClassifierDataset([texts[i] for i in idx_test], y[idx_test], tokenizer, config.MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=False)

    test_texts = [texts[i] for i in idx_test]
    test_labels = y[idx_test]

    return train_loader, val_loader, test_loader, idx_test, test_texts, test_labels


def train_classifier(
    model: CLModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> dict[str, list[float]]:
    print("\n[4/5] Classifier training on frozen encoder...")

    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.CLF_LR)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state = None
    best_val_loss = float("inf")
    patience = 3
    no_improve = 0

    for epoch in range(1, config.CLF_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        train_total_steps = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                emb = model.encode(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
            logits = model.classify(emb)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if step % config.LOG_EVERY_STEPS == 0 or step == train_total_steps:
                running_avg = float(np.mean(train_losses)) if train_losses else 0.0
                print(
                    f"    [CLF-Train][Epoch {epoch}/{config.CLF_EPOCHS}] "
                    f"step {step}/{train_total_steps} "
                    f"loss={float(loss.item()):.4f} running_avg={running_avg:.4f}"
                )

        model.eval()
        val_losses: list[float] = []
        val_preds: list[int] = []
        val_labels: list[int] = []
        val_total_steps = len(val_loader)

        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                batch = _to_device(batch, device)
                emb = model.encode(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
                logits = model.classify(emb)
                loss = criterion(logits, batch["labels"])
                val_losses.append(float(loss.item()))
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(batch["labels"].cpu().tolist())
                if step % config.LOG_EVERY_STEPS == 0 or step == val_total_steps:
                    running_avg = float(np.mean(val_losses)) if val_losses else 0.0
                    print(
                        f"    [CLF-Val][Epoch {epoch}/{config.CLF_EPOCHS}] "
                        f"step {step}/{val_total_steps} "
                        f"loss={float(loss.item()):.4f} running_avg={running_avg:.4f}"
                    )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = float(accuracy_score(val_labels, val_preds)) if val_labels else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"  Epoch {epoch}/{config.CLF_EPOCHS} - "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.classifier.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping triggered.")
                break

    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    return history


def evaluate_and_export(
    model: CLModel,
    test_loader: DataLoader,
    test_texts: list[str],
    test_labels_int: np.ndarray,
    label_encoder: LabelEncoder,
    config: Config,
    device: torch.device,
) -> tuple[float, str, str]:
    print("\n[5/5] Evaluation and export...")

    model.eval()

    y_pred: list[int] = []
    y_probs: list[list[float]] = []
    embeddings: list[list[float]] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = _to_device(batch, device)
            emb = model.encode(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            logits = model.classify(emb)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())
            embeddings.extend(emb.cpu().tolist())

    acc = float(accuracy_score(test_labels_int, y_pred))
    report_text = str(
        classification_report(
        test_labels_int,
        y_pred,
        target_names=list(label_encoder.classes_),
        digits=4,
        zero_division=0,
        )
    )

    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Accuracy: {acc:.4f}\n\n")
        handle.write(report_text)

    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Classification report saved -> {report_path}")

    handoff = []
    for i, text in enumerate(test_texts):
        handoff.append(
            {
                "text": text,
                "predicted_label": label_encoder.classes_[y_pred[i]],
                "true_label": label_encoder.classes_[test_labels_int[i]],
                "label_probs": {
                    cls: float(y_probs[i][j])
                    for j, cls in enumerate(label_encoder.classes_)
                },
                "embedding": embeddings[i],
            }
        )

    handoff_path = os.path.join(config.OUTPUT_DIR, "cl_outputs_for_fact_verification.json")
    with open(handoff_path, "w", encoding="utf-8") as handle:
        json.dump(handoff, handle, indent=2, ensure_ascii=False)

    print(f"  Handoff file saved -> {handoff_path}")

    weights_path = os.path.join(config.OUTPUT_DIR, "cl_model_weights.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_classes": list(label_encoder.classes_),
            "config": asdict(config),
        },
        weights_path,
    )
    print(f"  Model weights saved -> {weights_path}")

    return acc, report_text, handoff_path


def run_training(
    data_path: str,
    output_dir: str,
    config: Config | None = None,
    emotion_sim_threshold: float | None = None,
    use_gpu: bool = False,
) -> dict:
    config = config or Config()
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    if emotion_sim_threshold is not None:
        config.EMOTION_SIM_THRESHOLD = emotion_sim_threshold

    if not Path(config.DATA_PATH).exists():
        raise FileNotFoundError(f"Training data file not found: {config.DATA_PATH}")

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("--use-gpu was requested, but CUDA is not available in PyTorch.")

    device = torch.device("cuda" if use_gpu else "cpu")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    set_global_seed(config.RANDOM_SEED)

    pairs, all_samples = load_and_pair_data(config)
    if not pairs:
        raise ValueError("No valid contrastive pairs were found in the training data.")
    if not all_samples:
        raise ValueError("No classifier samples were found in the training data.")

    print("\n[2/5] Loading tokenizer and encoder...")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    encoder = AutoModel.from_pretrained(config.BERT_MODEL_NAME)

    label_encoder = LabelEncoder()
    label_encoder.fit(config.LABEL_NAMES)
    num_classes = len(label_encoder.classes_)

    pair_ds = ContrastivePairDataset(pairs, tokenizer, config.MAX_SEQ_LEN)
    cl_loader = DataLoader(pair_ds, batch_size=config.CL_BATCH_SIZE, shuffle=True)

    train_loader, val_loader, test_loader, _test_idx, test_texts, test_labels_int = build_classifier_loaders(
        all_samples,
        tokenizer,
        label_encoder,
        config,
    )

    model = CLModel(encoder, config.HIDDEN_DIM, config.PROJECTION_DIM, num_classes).to(device)

    cl_history = train_contrastive(model, cl_loader, config, device)
    clf_history = train_classifier(model, train_loader, val_loader, config, device)

    acc, report, handoff_path = evaluate_and_export(
        model,
        test_loader,
        test_texts,
        test_labels_int,
        label_encoder,
        config,
        device,
    )

    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cl_loss_per_epoch": cl_history,
                "clf_train_loss": clf_history.get("train_loss", []),
                "clf_val_loss": clf_history.get("val_loss", []),
                "clf_val_accuracy": clf_history.get("val_accuracy", []),
                "final_test_accuracy": acc,
            },
            handle,
            indent=2,
        )

    print("\n" + "=" * 65)
    print(f" Done. All outputs saved to -> {config.OUTPUT_DIR}/")
    print("   cl_outputs_for_fact_verification.json  -> handoff to Fact Verification")
    print("   cl_model_weights.pt                    -> saved model")
    print("   classification_report.txt              -> eval results")
    print("   training_history.json                  -> loss/accuracy curves")
    print("=" * 65)

    return {
        "data_path": config.DATA_PATH,
        "output_dir": config.OUTPUT_DIR,
        "test_accuracy": acc,
        "handoff_path": handoff_path,
        "history_path": history_path,
        "device": str(device),
    }


def main() -> None:
    print("=" * 65)
    print(" LACL — Contrastive Learning Module (PyTorch)")
    print("=" * 65)

    config = Config()
    run_training(
        data_path=config.DATA_PATH,
        output_dir=config.OUTPUT_DIR,
        config=config,
        use_gpu=False,
    )


if __name__ == "__main__":
    main()
