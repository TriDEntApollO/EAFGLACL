from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _is_likely_dedicated_gpu(device_name: str) -> bool:
    lowered = device_name.lower()
    integrated_markers = ("intel", "uhd", "iris")
    return not any(marker in lowered for marker in integrated_markers)


def _configure_training_device(use_gpu: bool) -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not use_gpu:
        # Keep default execution on CPU unless user explicitly opts in to GPU training.
        tf.config.set_visible_devices([], "GPU")
        return "CPU"

    if not gpus:
        raise SystemExit(
            "--use-gpu was requested, but no GPU was detected by TensorFlow."
        )

    dedicated_candidates = []
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        device_name = str(details.get("device_name", gpu.name))
        if _is_likely_dedicated_gpu(device_name):
            dedicated_candidates.append((gpu, device_name))

    if not dedicated_candidates:
        raise SystemExit(
            "--use-gpu was requested, but no dedicated GPU was found. "
            "Detected GPUs appear to be integrated only."
        )

    selected_gpu, selected_name = dedicated_candidates[0]
    tf.config.set_visible_devices([selected_gpu], "GPU")
    try:
        tf.config.experimental.set_memory_growth(selected_gpu, True)
    except Exception:
        # Memory growth may not be supported on all configurations.
        pass

    return f"GPU ({selected_name})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contrastive learning model training.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Datasets/complete_data.jsonl"),
        help="Input JSONL path for CL training (default: Datasets/complete_data.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cl_outputs"),
        help="Output directory for training artifacts (default: outputs/cl_outputs).",
    )
    parser.add_argument(
        "--emotion-sim-threshold",
        type=float,
        default=None,
        help="Optional override for minimum emotion similarity used for CL pairs.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Train on a dedicated GPU. If no dedicated GPU is detected, exit with an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device_mode = _configure_training_device(args.use_gpu)

    from models.cl_model import run_training

    result = run_training(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        emotion_sim_threshold=args.emotion_sim_threshold,
    )
    result["device_mode"] = device_mode
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
