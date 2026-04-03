from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.builder import build_unified_dataset
from data.inspection import write_inspection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified dataset for EAFGLACL.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".datasets_raw"),
        help="Root directory containing raw datasets (default: .datasets_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Datasets"),
        help="Output directory for processed files (default: Datasets).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional cap per dataset for fast smoke runs.",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Annotate each record with predicted emotion.",
    )
    parser.add_argument(
        "--enable-fact",
        action="store_true",
        help="Annotate each record with Wikipedia-based fact consistency score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inspection = write_inspection_report(args.data_root, args.output_dir / "inspection.json")
    stats = build_unified_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_dataset=args.max_per_dataset,
        enable_emotion=args.enable_emotion,
        enable_fact=args.enable_fact,
    )

    print(json.dumps({"inspection": inspection, "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
