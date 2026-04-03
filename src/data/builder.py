from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .enrichment import enrich_records
from .loaders_pheme import load_pheme_dataset
from .loaders_twitter import load_twitter_dataset
from .loaders_weibo import load_weibo_dataset
from .schema import UnifiedRecord


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stratified_split(
    records: list[UnifiedRecord], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[UnifiedRecord], list[UnifiedRecord], list[UnifiedRecord]]:
    random.seed(seed)
    by_label: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    train: list[UnifiedRecord] = []
    val: list[UnifiedRecord] = []
    test: list[UnifiedRecord] = []

    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


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
    weibo = load_weibo_dataset(data_root / "Weibo", max_records=max_per_dataset)
    pheme = load_pheme_dataset(data_root / "PHEME", max_records=max_per_dataset)

    unified = twitter15 + twitter16 + weibo + pheme
    unified = enrich_records(unified, enable_emotion=enable_emotion, enable_fact=enable_fact)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = _stratified_split(unified, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    _write_jsonl(output_dir / "unified_records.jsonl", (r.to_dict() for r in unified))
    _write_jsonl(output_dir / "train.jsonl", (r.to_dict() for r in train))
    _write_jsonl(output_dir / "val.jsonl", (r.to_dict() for r in val))
    _write_jsonl(output_dir / "test.jsonl", (r.to_dict() for r in test))

    stats = {
        "total_records": len(unified),
        "by_dataset": dict(Counter(r.dataset for r in unified)),
        "by_label": dict(Counter(r.label for r in unified)),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "avg_comment_count": (sum(len(r.comments) for r in unified) / len(unified)) if unified else 0.0,
        "emotion_enriched": enable_emotion,
        "fact_enriched": enable_fact,
        "max_per_dataset": max_per_dataset,
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return stats
