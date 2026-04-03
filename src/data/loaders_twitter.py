from __future__ import annotations

from pathlib import Path

from .normalization import clean_text, normalize_label
from .schema import UnifiedRecord


def _read_label_file(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            raw_label, tweet_id = line.split(":", maxsplit=1)
            labels[tweet_id.strip()] = normalize_label(raw_label.strip())
    return labels


def _read_source_tweets(path: Path) -> dict[str, str]:
    tweets: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            tweet_id, text = line.split("\t", maxsplit=1)
            tweets[tweet_id.strip()] = clean_text(text)
    return tweets


def load_twitter_dataset(
    dataset_root: Path, dataset_name: str, max_records: int | None = None
) -> list[UnifiedRecord]:
    label_path = dataset_root / "label.txt"
    source_path = dataset_root / "source_tweets.txt"

    if not label_path.exists() or not source_path.exists():
        raise FileNotFoundError(f"Missing expected files for {dataset_name}: {dataset_root}")

    labels = _read_label_file(label_path)
    sources = _read_source_tweets(source_path)
    records: list[UnifiedRecord] = []

    for tweet_id, text in sources.items():
        if tweet_id not in labels:
            continue
        records.append(
            UnifiedRecord(
                record_id=f"{dataset_name}:{tweet_id}",
                dataset=dataset_name,
                label=labels[tweet_id],
                text=text,
                thread_id=tweet_id,
                source_id=tweet_id,
                metadata={"source": "source_tweets.txt"},
            )
        )

        if max_records is not None and len(records) >= max_records:
            break

    return records
