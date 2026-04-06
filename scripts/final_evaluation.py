from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


DEFAULT_DATA_PATH = Path("Datasets/complete_data.jsonl")
DEFAULT_MODEL_WEIGHTS = Path("outputs/cl_outputs/cl_model_weights.pt")
LEGACY_TF_INDEX = Path("outputs/cl_outputs/cl_model_weights.index")
LEGACY_TF_DATA = Path("outputs/cl_outputs/cl_model_weights.data-00000-of-00001")
LEGACY_HANDOFF_PATH = Path("outputs/cl_outputs/cl_outputs_for_fact_verification.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final evaluation on CL outputs and report metrics "
            "with and without fact verification."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Input dataset JSONL path for live CL inference "
            "(default: Datasets/complete_data.jsonl)."
        ),
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS,
        help=(
            "Exported CL model weights checkpoint (.pt) "
            "(default: outputs/cl_outputs/cl_model_weights.pt)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/final_outputs/final_evaluation_records.json"),
        help="Output JSON with baseline + fact-enriched record predictions.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/final_outputs/final_evaluation_metrics.json"),
        help="Output JSON for aggregate final evaluation metrics.",
    )
    parser.add_argument(
        "--disable-fact-verification",
        action="store_true",
        help="Skip fact verification and report baseline-only metrics.",
    )
    parser.add_argument(
        "--fact-provider",
        choices=["gemini_plus_wikipedia", "wikipedia_only"],
        default="gemini_plus_wikipedia",
        help=(
            "Fact verification provider flow. "
            "`gemini_plus_wikipedia` uses Gemini claim workflow with Wikipedia evidence, "
            "`wikipedia_only` uses only Wikipedia verifier scoring."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run CL model inference on CUDA. Exits if CUDA is unavailable.",
    )
    return parser.parse_args()


def _load_dataset_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_handoff_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}.")
    return payload


def _run_live_cl_inference(
    data_path: Path,
    model_weights: Path,
    max_records: int | None,
    use_gpu: bool,
) -> tuple[list[dict[str, Any]], str]:
    from models.cl_model import CLModel

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not model_weights.exists():
        if LEGACY_TF_INDEX.exists() and LEGACY_TF_DATA.exists() and LEGACY_HANDOFF_PATH.exists():
            print(
                "Detected legacy TensorFlow CL checkpoints (.index/.data). "
                "Current evaluator uses PyTorch live inference, so using legacy handoff "
                "outputs/cl_outputs/cl_outputs_for_fact_verification.json as compatibility input."
            )
            rows = _load_handoff_json(LEGACY_HANDOFF_PATH)
            if max_records is not None:
                rows = rows[:max_records]
            if not rows:
                raise ValueError("No rows found in legacy handoff input.")
            return rows, "compatibility_handoff"

        raise FileNotFoundError(
            "Model weights file not found: "
            f"{model_weights}. "
            "Run training first to export a PyTorch checkpoint "
            "(outputs/cl_outputs/cl_model_weights.pt), or pass --model-weights with a valid .pt path."
        )

    if use_gpu and not torch.cuda.is_available():
        raise SystemExit("--use-gpu was requested, but CUDA is not available in this environment.")

    device = torch.device("cuda" if use_gpu else "cpu")
    checkpoint = torch.load(model_weights, map_location=device)

    label_classes = checkpoint.get("label_classes")
    if not isinstance(label_classes, list) or not label_classes:
        raise ValueError("Checkpoint is missing valid label_classes metadata.")

    cfg = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    bert_model_name = str(cfg.get("BERT_MODEL_NAME", "bert-base-uncased"))
    max_seq_len = int(cfg.get("MAX_SEQ_LEN", 128))
    hidden_dim = int(cfg.get("HIDDEN_DIM", 256))
    projection_dim = int(cfg.get("PROJECTION_DIM", 128))
    batch_size = int(cfg.get("CLF_BATCH_SIZE", 64))

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    encoder = AutoModel.from_pretrained(bert_model_name)
    model = CLModel(encoder, hidden_dim, projection_dim, len(label_classes)).to(device)
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    model.load_state_dict(model_state)
    model.eval()

    raw_rows = _load_dataset_jsonl(data_path)
    if max_records is not None:
        raw_rows = raw_rows[:max_records]

    if not raw_rows:
        raise ValueError("No rows found in dataset input.")

    texts: list[str] = []
    true_labels: list[str] = []
    for row in raw_rows:
        texts.append(str(row.get("text", "")).strip())
        true_labels.append(str(row.get("label", row.get("true_label", "")).strip()))

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
    )

    predicted_labels: list[str] = []
    probs_by_row: list[list[float]] = []
    total = len(texts)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_input_ids = encoded["input_ids"][start:end].to(device)
            batch_attention_mask = encoded["attention_mask"][start:end].to(device)
            batch_token_type_ids = (
                encoded["token_type_ids"][start:end].to(device)
                if "token_type_ids" in encoded
                else None
            )

            emb = model.encode(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            logits = model.classify(emb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            probs_cpu = probs.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            for i, pred_idx in enumerate(preds_cpu):
                predicted_labels.append(str(label_classes[pred_idx]))
                probs_by_row.append([float(x) for x in probs_cpu[i]])

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        out = dict(row)
        out["text"] = texts[i]
        out["true_label"] = true_labels[i]
        out["predicted_label"] = predicted_labels[i]
        out["label_probs"] = {
            cls: float(probs_by_row[i][j])
            for j, cls in enumerate(label_classes)
        }
        rows.append(out)

    return rows, str(device)


def _score_text(text: str, verifier: Any) -> float:
    if hasattr(verifier, "get_fact_score"):
        result = verifier.get_fact_score(text, verbose=False)
        return float(result.confidence)
    if hasattr(verifier, "fact_consistency_score"):
        return float(verifier.fact_consistency_score(text))
    raise TypeError("Verifier does not expose a supported scoring interface.")


def _map_fact_score_to_label(score: float) -> str:
    if score >= 0.66:
        return "true"
    if score <= 0.34:
        return "false"
    return "unverified"


def _final_label(predicted_label: str, fact_label: str) -> str:
    if predicted_label == "non-rumor":
        return "non-rumor"
    return fact_label


def run_final_evaluation(
    data_path: Path,
    model_weights: Path,
    output_path: Path,
    metrics_path: Path,
    max_records: int | None,
    disable_fact_verification: bool,
    fact_provider: str,
    use_gpu: bool,
) -> dict[str, Any]:
    verifier = None
    if not disable_fact_verification:
        if fact_provider == "wikipedia_only":
            from fact.wikipedia_verifier import WikipediaFactVerifier

            verifier = WikipediaFactVerifier()
        else:
            from fact.gemini_verifier import GeminiFactVerifier

            verifier = GeminiFactVerifier()

    rows, device_name = _run_live_cl_inference(
        data_path=data_path,
        model_weights=model_weights,
        max_records=max_records,
        use_gpu=use_gpu,
    )

    print(f"Loaded {len(rows)} records from {data_path}")
    print(f"Live CL inference device: {device_name}")
    print(f"Model checkpoint: {model_weights}")
    if disable_fact_verification:
        print("Fact verification: disabled (baseline-only)")
    else:
        if fact_provider == "wikipedia_only":
            print("Fact verification flow: Wikipedia-only verification")
        else:
            print("Fact verification flow: Gemini claim extraction + Wikipedia evidence + Gemini verification")

    enriched: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred_baseline: list[str] = []
    y_pred_final: list[str] = []

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        true_label = str(row.get("true_label", "")).strip()
        predicted_label = str(row.get("predicted_label", "")).strip()

        if disable_fact_verification:
            fact_score = None
            fact_label = predicted_label
            final_label = predicted_label
        else:
            fact_score = _score_text(text, verifier)
            fact_label = _map_fact_score_to_label(fact_score)
            final_label = _final_label(predicted_label, fact_label)

        y_true.append(true_label)
        y_pred_baseline.append(predicted_label)
        y_pred_final.append(final_label)

        out_row = dict(row)
        out_row["fact_score"] = fact_score
        out_row["fact_label"] = fact_label
        out_row["final_label"] = final_label
        out_row["fact_verification_applied"] = not disable_fact_verification
        enriched.append(out_row)

        if idx % 50 == 0 or idx == total:
            print(f"  Processed {idx}/{total} records")

    baseline_acc = float(accuracy_score(y_true, y_pred_baseline))
    final_acc = float(accuracy_score(y_true, y_pred_final))

    rumor_indices = [i for i, label in enumerate(y_true) if label != "non-rumor"]
    if rumor_indices:
        rumor_true = [y_true[i] for i in rumor_indices]
        rumor_baseline = [y_pred_baseline[i] for i in rumor_indices]
        rumor_final = [y_pred_final[i] for i in rumor_indices]
        rumor_baseline_acc = float(accuracy_score(rumor_true, rumor_baseline))
        rumor_final_acc = float(accuracy_score(rumor_true, rumor_final))
    else:
        rumor_baseline_acc = 0.0
        rumor_final_acc = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, ensure_ascii=False, indent=2)

    metrics = {
        "data_path": str(data_path),
        "model_weights": str(model_weights),
        "output_path": str(output_path),
        "inference_device": device_name,
        "fact_verification_enabled": not disable_fact_verification,
        "fact_verification_flow": (
            fact_provider
            if not disable_fact_verification
            else None
        ),
        "records_evaluated": len(rows),
        "accuracy_without_fact_verification": baseline_acc,
        "accuracy_with_current_pipeline": final_acc,
        "rumor_only_accuracy_without_fact_verification": rumor_baseline_acc,
        "rumor_only_accuracy_with_current_pipeline": rumor_final_acc,
        "accuracy_delta": final_acc - baseline_acc,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\nFinal evaluation completed.")
    print(f"  Accuracy without fact verification: {baseline_acc:.4f}")
    print(f"  Accuracy with current pipeline: {final_acc:.4f}")
    print(f"  Delta: {final_acc - baseline_acc:+.4f}")
    print(f"  Saved evaluated records -> {output_path}")
    print(f"  Saved metrics -> {metrics_path}")

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_final_evaluation(
        data_path=args.data_path,
        model_weights=args.model_weights,
        output_path=args.output_path,
        metrics_path=args.metrics_path,
        max_records=args.max_records,
        disable_fact_verification=args.disable_fact_verification,
        fact_provider=args.fact_provider,
        use_gpu=args.use_gpu,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
