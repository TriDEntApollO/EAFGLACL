from __future__ import annotations

import argparse
import json
import subprocess
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


def _detect_nvidia_gpu_names() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if completed.returncode != 0:
        return []

    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return names


def _configure_training_device(use_gpu: bool) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in the active environment. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if not use_gpu:
        return "CPU"

    if not torch.cuda.is_available():
        nvidia_names = _detect_nvidia_gpu_names()
        details = ["--use-gpu was requested, but CUDA is not available in this PyTorch runtime."]
        if nvidia_names:
            details.append(
                "Detected NVIDIA GPU hardware via nvidia-smi: " + ", ".join(nvidia_names)
            )
            details.append(
                "Install a CUDA-enabled PyTorch build matching your driver/CUDA setup."
            )
        raise SystemExit(" ".join(details))

    device_name = torch.cuda.get_device_name(0)
    if not _is_likely_dedicated_gpu(device_name):
        raise SystemExit(
            "--use-gpu was requested, but detected CUDA device appears integrated: "
            f"{device_name}"
        )

    return f"GPU ({device_name})"


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
        use_gpu=args.use_gpu,
    )
    result["device_mode"] = device_mode
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
