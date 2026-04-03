from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.inspection import write_inspection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset folder layout.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".datasets_raw"),
        help="Root directory containing raw datasets (default: .datasets_raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Datasets/inspection.json"),
        help="Output JSON report path (default: Datasets/inspection.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = write_inspection_report(args.data_root, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
