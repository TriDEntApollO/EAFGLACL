from __future__ import annotations

import json
from pathlib import Path


def inspect_dataset_layout(data_root: Path) -> dict:
    report: dict[str, dict] = {}

    twitter15 = data_root / "twitter15"
    twitter16 = data_root / "twitter16"
    rumoureval = data_root / "RumourEval"
    pheme = data_root / "PHEME"

    report["twitter15"] = {
        "exists": twitter15.exists(),
        "label_file": (twitter15 / "label.txt").exists(),
        "source_file": (twitter15 / "source_tweets.txt").exists(),
    }
    report["twitter16"] = {
        "exists": twitter16.exists(),
        "label_file": (twitter16 / "label.txt").exists(),
        "source_file": (twitter16 / "source_tweets.txt").exists(),
    }

    rumoureval_train = rumoureval / "train"
    rumoureval_test = rumoureval / "test"
    rumoureval_train_alt = rumoureval / "rumoureval-2019-training-data"
    rumoureval_test_alt = rumoureval / "rumoureval-2019-test-data"
    rumoureval_unified = rumoureval / "rumoureval-2019-data"

    if rumoureval_unified.exists():
        detected_train = rumoureval_unified
        detected_test = rumoureval_unified
    else:
        detected_train = rumoureval_train if rumoureval_train.exists() else rumoureval_train_alt
        detected_test = rumoureval_test if rumoureval_test.exists() else rumoureval_test_alt

    report["rumoureval"] = {
        "exists": rumoureval.exists(),
        "unified_root_exists": rumoureval_unified.exists(),
        "unified_root": str(rumoureval_unified.name) if rumoureval_unified.exists() else None,
        "train_exists": detected_train.exists(),
        "test_exists": detected_test.exists(),
        "train_root": str(detected_train.name) if detected_train.exists() else None,
        "test_root": str(detected_test.name) if detected_test.exists() else None,
        "train_json_files": len([p for p in detected_train.rglob("*.json") if p.is_file()])
        if detected_train.exists()
        else 0,
        "test_json_files": len([p for p in detected_test.rglob("*.json") if p.is_file()])
        if detected_test.exists()
        else 0,
    }

    pheme_root = pheme / "all-rnr-annotated-threads"
    event_dirs = [
        p.name for p in pheme_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ] if pheme_root.exists() else []
    report["pheme"] = {
        "exists": pheme.exists(),
        "event_count": len(event_dirs),
        "events": sorted(event_dirs),
    }

    return report


def write_inspection_report(data_root: Path, output_path: Path) -> dict:
    report = inspect_dataset_layout(data_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
