from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .batch_processor import BatchProcessor


def _load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def _save_jsonl(data: list[dict[str, Any]], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as handle:
        for record in data:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_emotion_filter_pipeline(
    augmented_input_path: Path,
    unified_input_path: Path,
    filtered_output_path: Path,
    complete_output_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    if not augmented_input_path.exists():
        raise FileNotFoundError(f"Augmented input file not found: {augmented_input_path}")
    if not unified_input_path.exists():
        raise FileNotFoundError(f"Unified input file not found: {unified_input_path}")

    processor = BatchProcessor()
    processor.run(str(augmented_input_path), str(filtered_output_path))

    unified_data = _load_jsonl(unified_input_path)
    filtered_data = _load_jsonl(filtered_output_path)

    complete_data = unified_data + filtered_data
    random.Random(seed).shuffle(complete_data)
    _save_jsonl(complete_data, complete_output_path)

    stats = {
        "augmented_input_path": str(augmented_input_path),
        "unified_input_path": str(unified_input_path),
        "filtered_output_path": str(filtered_output_path),
        "complete_output_path": str(complete_output_path),
        "original_records": len(unified_data),
        "filtered_augmented_records": len(filtered_data),
        "complete_records": len(complete_data),
        "shuffle_seed": seed,
    }
    return stats
