"""
=============================================================================
LACL - Contrastive Learning Module
=============================================================================
Project     : LLM-Augmented Contrastive Learning for Misinformation Detection
Module      : Contrastive Learning (CL) — Stage 4 of pipeline
Input       : Final Dataset (Complete_data.jsonl) from emotion filtering stage
Output      : Trained model weights + embeddings + predicted labels JSON
              → handed off to Fact Verification stage

Architecture (follows original LACL paper):
  - Encoder  : BERT-base-uncased (frozen after CL training)
  - Projector: 2-layer MLP → projection space for contrastive loss
  - Loss     : InfoNCE (NT-Xent) contrastive loss with temperature τ
  - Classifier: Fully connected + Softmax (trained on frozen encoder)

Labels      : non-rumor, false, true, unverified (4-class)
Framework   : TensorFlow 2.x / tf-keras
=============================================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, TFBertModel


# =============================================================================
# CONFIG — change these as needed
# =============================================================================

class Config:
    # Paths
    """
    If you face issue with dataset path replace -->
    DATA_PATH        = "Complete_data.jsonl"   to 
    "
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/emotion
    base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))  # .../EAFGLACL

    dataset_dir = os.path.join(base_dir, 'Dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    DATA_PATH        = os.path.join(dataset_dir, 'Complete_data.jsonl')
    "
    """
    DATA_PATH        = "Complete_data.jsonl"          # input dataset
    OUTPUT_DIR       = "cl_outputs"                   # all outputs saved here

    # Model
    BERT_MODEL_NAME  = "bert-base-uncased"            # encoder backbone
    MAX_SEQ_LEN      = 128                            # max token length
    PROJECTION_DIM   = 128                            # MLP projection head output dim
    HIDDEN_DIM       = 256                            # MLP hidden layer dim

    # Contrastive Learning
    TEMPERATURE      = 0.07                           # τ in InfoNCE loss
    CL_EPOCHS        = 5                              # contrastive pre-training epochs
    CL_BATCH_SIZE    = 32                             # pairs per batch
    CL_LR            = 2e-5                           # learning rate for CL phase

    # Classifier
    CLF_EPOCHS       = 10                             # classifier fine-tuning epochs
    CLF_BATCH_SIZE   = 64                             # batch size for classifier
    CLF_LR           = 1e-4                           # learning rate for classifier

    # Emotion similarity threshold (from emotion filtering stage)
    EMOTION_SIM_THRESHOLD = 0.85                      # only use pairs above this

    # Train/Val/Test split
    VAL_SIZE         = 0.1
    TEST_SIZE        = 0.2
    RANDOM_SEED      = 42

    # Labels
    LABEL_NAMES      = ["non-rumor", "false", "true", "unverified"]


# =============================================================================
# STEP 1: DATA LOADING & PAIRING
# =============================================================================

def load_and_pair_data(config: Config):
    """
    Reads the JSONL file and creates (original, augmented) positive pairs
    for contrastive learning.

    Returns:
        pairs       : list of (orig_text, aug_text, label) tuples
        all_samples : list of (text, label) for classifier training
    """
    print("\n[1/5] Loading dataset...")

    originals  = {}   # record_id -> record
    augmented  = []   # list of augmented records

    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
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

    # Build (original, augmented) positive pairs
    # Each original can have multiple augmented versions → use all pairs
    pairs = []
    skipped = 0

    for aug in augmented:
        orig_id = aug.get("source_record_id")
        if orig_id not in originals:
            skipped += 1
            continue

        # Emotion similarity filter — only high-quality pairs
        em_sim = aug.get("emotion_similarity", 1.0)
        if em_sim < config.EMOTION_SIM_THRESHOLD:
            skipped += 1
            continue

        orig = originals[orig_id]
        orig_text = orig.get("text", "").strip()
        aug_text  = aug.get("text", "").strip()
        label     = orig.get("label", aug.get("label", "non-rumor"))

        if orig_text and aug_text:
            pairs.append((orig_text, aug_text, label))

    # All unique original samples for classifier training
    all_samples = [
        (rec["text"].strip(), rec["label"])
        for rec in originals.values()
        if rec.get("text", "").strip() and rec.get("label")
    ]

    print(f"  Valid pairs: {len(pairs)}  (skipped: {skipped})")
    print(f"  All samples for classifier: {len(all_samples)}")

    # Label distribution
    from collections import Counter
    label_dist = Counter(p[2] for p in pairs)
    print(f"  Label distribution in pairs: {dict(label_dist)}")

    return pairs, all_samples


# =============================================================================
# STEP 2: TOKENIZATION
# =============================================================================

def tokenize_texts(texts, tokenizer, max_len):
    """
    Tokenizes a list of strings using the BERT tokenizer.

    Returns dict with 'input_ids', 'attention_mask', 'token_type_ids'
    each as numpy arrays of shape (N, max_len).
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return {
        "input_ids"      : enc["input_ids"],
        "attention_mask" : enc["attention_mask"],
        "token_type_ids" : enc.get("token_type_ids",
                           np.zeros_like(enc["input_ids"])),
    }


def build_pair_dataset(pairs, tokenizer, config: Config):
    """
    Tokenizes all positive pairs and returns TF datasets for CL training.

    Each batch contains:
        orig_input_ids, orig_attention_mask, orig_token_type_ids,
        aug_input_ids,  aug_attention_mask,  aug_token_type_ids
    No labels needed for contrastive pre-training.
    """
    orig_texts = [p[0] for p in pairs]
    aug_texts  = [p[1] for p in pairs]

    print("  Tokenizing original texts...")
    orig_enc = tokenize_texts(orig_texts, tokenizer, config.MAX_SEQ_LEN)
    print("  Tokenizing augmented texts...")
    aug_enc  = tokenize_texts(aug_texts,  tokenizer, config.MAX_SEQ_LEN)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "orig_input_ids"       : orig_enc["input_ids"],
            "orig_attention_mask"  : orig_enc["attention_mask"],
            "orig_token_type_ids"  : orig_enc["token_type_ids"],
            "aug_input_ids"        : aug_enc["input_ids"],
            "aug_attention_mask"   : aug_enc["attention_mask"],
            "aug_token_type_ids"   : aug_enc["token_type_ids"],
        }
    ))

    dataset = (
        dataset
        .shuffle(buffer_size=2048, seed=config.RANDOM_SEED)
        .batch(config.CL_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def build_classifier_dataset(samples, tokenizer, label_encoder, config: Config):
    """
    Tokenizes all original samples for classifier training.
    Returns train/val/test splits as TF datasets.
    """
    texts  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # Encode labels to integers
    y = label_encoder.transform(labels)

    print("  Tokenizing classifier samples...")
    enc = tokenize_texts(texts, tokenizer, config.MAX_SEQ_LEN)

    X = {
        "input_ids"      : enc["input_ids"],
        "attention_mask" : enc["attention_mask"],
        "token_type_ids" : enc["token_type_ids"],
    }

    # Split into train / val / test
    indices = np.arange(len(texts))
    idx_train, idx_test = train_test_split(
        indices, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_SEED, stratify=y[idx_train]
    )

    def make_ds(idx, shuffle=False):
        x = {k: v[idx] for k, v in X.items()}
        y_split = y[idx]
        ds = tf.data.Dataset.from_tensor_slices((x, y_split))
        if shuffle:
            ds = ds.shuffle(buffer_size=1024, seed=config.RANDOM_SEED)
        return ds.batch(config.CLF_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return (
        make_ds(idx_train, shuffle=True),
        make_ds(idx_val),
        make_ds(idx_test),
        idx_test,
        y[idx_test],
        [texts[i] for i in idx_test],
    )


# =============================================================================
# STEP 3: MODEL ARCHITECTURE
# =============================================================================

class ProjectionHead(keras.layers.Layer):
    """
    2-layer MLP projection head as described in the LACL paper.
    Maps BERT [CLS] embeddings → projection space for InfoNCE loss.

    zi = W2 * σ(W1 * fi)
    """

    def __init__(self, hidden_dim: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(hidden_dim, activation="relu",
                                         name="proj_hidden")
        self.dense2 = keras.layers.Dense(projection_dim, name="proj_output")
        self.bn     = keras.layers.BatchNormalization(name="proj_bn")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dense2(x)
        return x


class CLModel(keras.Model):
    """
    Full Contrastive Learning model:
        BERT encoder  →  [CLS] embedding
        ProjectionHead → zi (used for InfoNCE loss during CL training)
        Classifier    → softmax label prediction (trained after CL)

    Encoder is SHARED between original and augmented inputs.
    """

    def __init__(self, bert_model, config: Config, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.bert       = bert_model
        self.projector  = ProjectionHead(config.HIDDEN_DIM,
                                         config.PROJECTION_DIM,
                                         name="projector")
        self.classifier = keras.layers.Dense(num_classes, activation="softmax",
                                             name="classifier")

    def encode(self, input_ids, attention_mask, token_type_ids, training=False):
        """Pass inputs through BERT and return [CLS] token embedding."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )
        # [CLS] embedding — shape (batch, 768)
        return outputs.last_hidden_state[:, 0, :]

    def project(self, embedding, training=False):
        """Map embedding to projection space — shape (batch, projection_dim)."""
        return self.projector(embedding, training=training)

    def classify(self, embedding):
        """Classify from frozen embedding — shape (batch, num_classes)."""
        return self.classifier(embedding)

    def call(self, inputs, training=False):
        """Used during classifier training — single input."""
        f = self.encode(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            training=training,
        )
        return self.classify(f)


# =============================================================================
# STEP 4: INFONCE CONTRASTIVE LOSS
# =============================================================================

def infonce_loss(z_orig, z_aug, temperature: float):
    """
    InfoNCE (NT-Xent) contrastive loss as defined in the LACL paper:

        L = -1/(2I) * Σ log [
              exp(sim(zi, zi') / τ)
              ─────────────────────────────────────────
              Σ_{j≠i} exp(sim(zi, zj) / τ)
            ]

    Positive pair : (z_orig[i], z_aug[i])  — same instance, different expression
    Negative pairs: all other samples in the batch

    Args:
        z_orig      : projected embeddings of originals  (batch, proj_dim)
        z_aug       : projected embeddings of augmented  (batch, proj_dim)
        temperature : τ hyperparameter

    Returns:
        scalar loss value
    """
    batch_size = tf.shape(z_orig)[0]

    # L2 normalize → cosine similarity via dot product
    z_orig = tf.math.l2_normalize(z_orig, axis=1)   # (B, D)
    z_aug  = tf.math.l2_normalize(z_aug,  axis=1)   # (B, D)

    # Concatenate: [orig; aug] → (2B, D)
    z_all  = tf.concat([z_orig, z_aug], axis=0)      # (2B, D)

    # Full similarity matrix (2B, 2B)
    sim_matrix = tf.matmul(z_all, z_all, transpose_b=True) / temperature

    # Mask out self-similarity on diagonal
    mask_diag = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(mask_diag, tf.fill(tf.shape(sim_matrix), -1e9), sim_matrix)

    # Positive pair indices:
    # for orig[i] → aug[i]  : positive at index (i, B+i)
    # for aug[i]  → orig[i] : positive at index (B+i, i)
    labels_orig = tf.range(batch_size, 2 * batch_size)   # B → 2B-1
    labels_aug  = tf.range(batch_size)                   # 0 → B-1
    labels      = tf.concat([labels_orig, labels_aug], axis=0)  # (2B,)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=sim_matrix,
    )

    return tf.reduce_mean(loss)


# =============================================================================
# STEP 5: TRAINING LOOPS
# =============================================================================

@tf.function
def cl_train_step(model, batch, optimizer, temperature):
    """Single contrastive learning training step."""
    with tf.GradientTape() as tape:
        # Encode both views
        f_orig = model.encode(
            batch["orig_input_ids"],
            batch["orig_attention_mask"],
            batch["orig_token_type_ids"],
            training=True,
        )
        f_aug = model.encode(
            batch["aug_input_ids"],
            batch["aug_attention_mask"],
            batch["aug_token_type_ids"],
            training=True,
        )
        # Project to contrastive space
        z_orig = model.project(f_orig, training=True)
        z_aug  = model.project(f_aug,  training=True)

        # InfoNCE loss
        loss = infonce_loss(z_orig, z_aug, temperature)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_contrastive(model, dataset, config: Config):
    """
    Phase 1: Contrastive pre-training.
    Trains BERT encoder + projection head using InfoNCE loss.
    Classifier weights are NOT updated here.
    """
    print("\n[3/5] Contrastive Pre-Training (Phase 1)...")

    optimizer = keras.optimizers.Adam(learning_rate=config.CL_LR)
    history   = []

    for epoch in range(config.CL_EPOCHS):
        epoch_losses = []

        for step, batch in enumerate(dataset):
            print(f"Total Epochs: {config.CL_EPOCHS}, Total Steps: {len(dataset)}, Current Epoch: {epoch+1}, Current Step: {step+1}")
            loss = cl_train_step(model, batch, optimizer, config.TEMPERATURE)
            epoch_losses.append(float(loss))

            if step % 20 == 0:
                print(f"  Epoch {epoch+1}/{config.CL_EPOCHS} "
                      f"| Step {step} | Loss: {float(loss):.4f}")

        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)
        print(f"  → Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    print("  Contrastive pre-training done.")
    return history


def train_classifier(model, train_ds, val_ds, config: Config, num_classes: int):
    """
    Phase 2: Classifier training on frozen encoder.
    Freezes BERT + projector, trains only the classifier head.
    """
    print("\n[4/5] Classifier Training (Phase 2)...")

    # Freeze encoder and projector
    model.bert.trainable      = False
    model.projector.trainable = False
    model.classifier.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.CLF_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.CLF_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# =============================================================================
# STEP 6: EVALUATION & OUTPUT GENERATION
# =============================================================================

def evaluate_and_export(model, test_ds, test_texts, test_labels_int,
                         label_encoder, config: Config, tokenizer):
    """
    Evaluates on test set and exports:
        1. embeddings + predictions JSON   → for Fact Verification stage
        2. classification report           → printed + saved
        3. model weights                   → saved to OUTPUT_DIR
    """
    print("\n[5/5] Evaluation & Export...")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Predictions ---
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(test_labels_int, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    report = classification_report(
        test_labels_int, y_pred,
        target_names=label_encoder.classes_,
        digits=4,
    )
    print(report)

    # Save report
    with open(os.path.join(config.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # --- Extract embeddings from frozen encoder ---
    print("  Extracting embeddings for handoff...")

    # Re-tokenize test texts to get embeddings
    enc = tokenize_texts(test_texts, tokenizer, config.MAX_SEQ_LEN)
    test_enc_ds = tf.data.Dataset.from_tensor_slices(enc).batch(64)

    embeddings = []
    for batch in test_enc_ds:
        f = model.encode(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            training=False,
        )
        embeddings.append(f.numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # --- Build handoff JSON for Fact Verification ---
    handoff = []
    for i, text in enumerate(test_texts):
        handoff.append({
            "text"            : text,
            "predicted_label" : label_encoder.classes_[y_pred[i]],
            "true_label"      : label_encoder.classes_[test_labels_int[i]],
            "label_probs"     : {
                cls: float(y_pred_probs[i][j])
                for j, cls in enumerate(label_encoder.classes_)
            },
            "embedding"       : embeddings[i].tolist(),   # 768-dim BERT [CLS]
        })

    handoff_path = os.path.join(config.OUTPUT_DIR, "cl_outputs_for_fact_verification.json")
    with open(handoff_path, "w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2, ensure_ascii=False)

    print(f"  Handoff file saved → {handoff_path}")
    print(f"  Records in handoff : {len(handoff)}")
    print(f"  Embedding shape    : {embeddings.shape}")

    # --- Save model weights ---
    weights_path = os.path.join(config.OUTPUT_DIR, "cl_model_weights")
    model.save_weights(weights_path)
    print(f"  Model weights saved → {weights_path}")

    return acc, report, handoff_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print(" LACL — Contrastive Learning Module")
    print("=" * 65)

    config = Config()
    run_training(data_path=config.DATA_PATH, output_dir=config.OUTPUT_DIR, config=config)


def run_training(
    data_path: str,
    output_dir: str,
    config: Config | None = None,
    emotion_sim_threshold: float | None = None,
) -> dict:
    config = config or Config()
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    if emotion_sim_threshold is not None:
        config.EMOTION_SIM_THRESHOLD = emotion_sim_threshold

    if not Path(config.DATA_PATH).exists():
        raise FileNotFoundError(f"Training data file not found: {config.DATA_PATH}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── 1. Load & pair data ──────────────────────────────────────────
    pairs, all_samples = load_and_pair_data(config)

    # ── 2. Setup tokenizer & encoder ─────────────────────────────────
    print("\n[2/5] Loading BERT tokenizer & model...")
    tokenizer  = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    bert_model = TFBertModel.from_pretrained(config.BERT_MODEL_NAME)
    print(f"  BERT loaded: {config.BERT_MODEL_NAME}")

    # ── 3. Label encoding ────────────────────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit(config.LABEL_NAMES)
    num_classes   = len(label_encoder.classes_)
    print(f"  Classes ({num_classes}): {list(label_encoder.classes_)}")

    # ── 4. Build TF datasets ─────────────────────────────────────────
    print("\n  Building CL pair dataset...")
    cl_dataset = build_pair_dataset(pairs, tokenizer, config)

    print("  Building classifier dataset...")
    (train_ds, val_ds, test_ds,
     test_idx, test_labels_int,
     test_texts) = build_classifier_dataset(
        all_samples, tokenizer, label_encoder, config
    )

    # ── 5. Build model ───────────────────────────────────────────────
    model = CLModel(bert_model, config, num_classes, name="LACL_CL")

    # ── 6. Phase 1: Contrastive pre-training ─────────────────────────
    cl_history = train_contrastive(model, cl_dataset, config)

    # ── 7. Phase 2: Classifier training ──────────────────────────────
    clf_history = train_classifier(model, train_ds, val_ds, config, num_classes)

    # ── 8. Evaluate & export ─────────────────────────────────────────
    acc, report, handoff_path = evaluate_and_export(
        model, test_ds, test_texts, test_labels_int,
        label_encoder, config, tokenizer
    )

    # ── 9. Save training history ─────────────────────────────────────
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "cl_loss_per_epoch"       : cl_history,
            "clf_val_accuracy"        : clf_history.history.get("val_accuracy", []),
            "clf_val_loss"            : clf_history.history.get("val_loss", []),
            "final_test_accuracy"     : acc,
        }, f, indent=2)

    print("\n" + "=" * 65)
    print(f" Done. All outputs saved to → {config.OUTPUT_DIR}/")
    print(f"   cl_outputs_for_fact_verification.json  ← handoff to Fact Verification")
    print(f"   cl_model_weights.*                     ← saved model")
    print(f"   classification_report.txt              ← eval results")
    print(f"   training_history.json                  ← loss/accuracy curves")
    print("=" * 65)

    return {
        "data_path": config.DATA_PATH,
        "output_dir": config.OUTPUT_DIR,
        "test_accuracy": acc,
        "handoff_path": handoff_path,
        "history_path": history_path,
    }


if __name__ == "__main__":
    main()
