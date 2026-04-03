from __future__ import annotations

import json
from pathlib import Path


def inspect_dataset_layout(data_root: Path) -> dict:
    report: dict[str, dict] = {}

    twitter15 = data_root / "twitter15"
    twitter16 = data_root / "twitter16"
    weibo = data_root / "Weibo"
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

    weibo_fake = weibo / "fake_news"
    weibo_real = weibo / "real_news"
    report["weibo"] = {
        "exists": weibo.exists(),
        "fake_news_dirs": len([p for p in weibo_fake.iterdir() if p.is_dir()]) if weibo_fake.exists() else 0,
        "real_news_dirs": len([p for p in weibo_real.iterdir() if p.is_dir()]) if weibo_real.exists() else 0,
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
