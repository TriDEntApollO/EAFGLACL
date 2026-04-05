from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from models.cl_model import run_training

    result = run_training(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        emotion_sim_threshold=args.emotion_sim_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
