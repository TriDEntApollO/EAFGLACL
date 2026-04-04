from __future__ import annotations

import json
import re
from pathlib import Path

from .normalization import clean_text, normalize_label
from .schema import UnifiedRecord


TEXT_KEYS = (
    "text",
    "tweet",
    "content",
    "body",
    "source_tweet",
    "source_text",
)

THREAD_ID_PATTERN = re.compile(r"^[0-9A-Za-z]+$")


def _safe_json_load(path: Path) -> dict | list | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _extract_text(payload: dict) -> str:
    for key in TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)

    # Reddit RumourEval source files are often wrapped as a Listing with children[0].data.
    if payload.get("kind") == "Listing":
        data = payload.get("data")
        if isinstance(data, dict):
            children = data.get("children")
            if isinstance(children, list) and children:
                first = children[0]
                if isinstance(first, dict):
                    child_data = first.get("data")
                    if isinstance(child_data, dict):
                        for key in ("selftext", "title", "text"):
                            value = child_data.get(key)
                            if isinstance(value, str) and value.strip():
                                return clean_text(value)
    return ""


def _load_subtaskb_labels(dataset_root: Path) -> dict[str, str]:
    key_path = dataset_root / "final-eval-key.json"
    payload = _safe_json_load(key_path)
    if not isinstance(payload, dict):
        return {}

    subtaskb = payload.get("subtaskbenglish")
    if not isinstance(subtaskb, dict):
        return {}

    labels: dict[str, str] = {}
    for thread_id, raw_label in subtaskb.items():
        if isinstance(thread_id, str) and isinstance(raw_label, str):
            labels[thread_id] = normalize_label(raw_label)
    return labels


def _iter_thread_dirs(partition_root: Path) -> list[Path]:
    thread_dirs: list[Path] = []
    for source_dir in partition_root.rglob("source-tweet"):
        if not source_dir.is_dir():
            continue
        thread_dir = source_dir.parent
        if THREAD_ID_PATTERN.fullmatch(thread_dir.name):
            thread_dirs.append(thread_dir)
    return sorted(thread_dirs)


def _resolve_source_file(thread_dir: Path) -> Path | None:
    source_dir = thread_dir / "source-tweet"
    preferred = source_dir / f"{thread_dir.name}.json"
    if preferred.exists():
        return preferred

    candidates = sorted([p for p in source_dir.glob("*.json") if p.is_file() and not p.name.startswith("._")])
    if candidates:
        return candidates[0]
    return None


def _resolve_partition_roots(dataset_root: Path) -> list[tuple[str, Path]]:
    candidates = [
        ("train", dataset_root / "train"),
        ("test", dataset_root / "test"),
        ("train", dataset_root / "rumoureval-2019-training-data"),
        ("test", dataset_root / "rumoureval-2019-test-data"),
        ("unified", dataset_root / "rumoureval-2019-data"),
    ]

    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for split, path in candidates:
        if path.exists() and path.is_dir() and path not in seen:
            roots.append((split, path))
            seen.add(path)
    return roots


def load_rumoureval_dataset(dataset_root: Path, max_records: int | None = None) -> list[UnifiedRecord]:
    records: list[UnifiedRecord] = []
    label_map = _load_subtaskb_labels(dataset_root)

    for split, split_root in _resolve_partition_roots(dataset_root):

        for thread_dir in _iter_thread_dirs(split_root):
            thread_id = thread_dir.name
            if thread_id not in label_map:
                continue

            file_path = _resolve_source_file(thread_dir)
            if file_path is None:
                continue

            payload = _safe_json_load(file_path)
            if not isinstance(payload, dict):
                continue

            text = _extract_text(payload)
            if not text:
                continue

            source_id = str(payload.get("id") or payload.get("id_str") or thread_id)
            label = label_map[thread_id]

            records.append(
                UnifiedRecord(
                    record_id=f"rumoureval:{split}:{thread_id}",
                    dataset="rumoureval",
                    label=label,
                    text=text,
                    language=payload.get("lang") or "en",
                    thread_id=thread_id,
                    source_id=source_id,
                    metadata={
                        "split": split,
                        "source_file": str(file_path.relative_to(dataset_root)),
                        "label_source": "final-eval-key.json:subtaskbenglish",
                    },
                )
            )

            if max_records is not None and len(records) >= max_records:
                return records

    return records
