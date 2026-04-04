from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .enrichment import enrich_records
from .loaders_pheme import load_pheme_dataset
from .loaders_rumoureval import load_rumoureval_dataset
from .loaders_twitter import load_twitter_dataset
from .schema import UnifiedRecord


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_unified_dataset(
    data_root: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_per_dataset: int | None = None,
    enable_emotion: bool = False,
    enable_fact: bool = False,
) -> dict:
    twitter15 = load_twitter_dataset(data_root / "twitter15", "twitter15", max_records=max_per_dataset)
    twitter16 = load_twitter_dataset(data_root / "twitter16", "twitter16", max_records=max_per_dataset)
    rumoureval = load_rumoureval_dataset(data_root / "RumourEval", max_records=max_per_dataset)
    pheme = load_pheme_dataset(data_root / "PHEME", max_records=max_per_dataset)

    unified = twitter15 + twitter16 + rumoureval + pheme
    unified = enrich_records(unified, enable_emotion=enable_emotion, enable_fact=enable_fact)
    random.Random(seed).shuffle(unified)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "unified_records.jsonl", (r.to_dict() for r in unified))

    stats = {
        "total_records": len(unified),
        "by_dataset": dict(Counter(r.dataset for r in unified)),
        "by_label": dict(Counter(r.label for r in unified)),
        "avg_comment_count": (sum(len(r.comments) for r in unified) / len(unified)) if unified else 0.0,
        "emotion_enriched": enable_emotion,
        "fact_enriched": enable_fact,
        "max_per_dataset": max_per_dataset,
        "shuffled": True,
        "shuffle_seed": seed,
        "split_deferred": True,
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return stats
