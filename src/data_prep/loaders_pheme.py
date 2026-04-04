from __future__ import annotations

import json
from pathlib import Path

from .normalization import clean_text, normalize_label
from .schema import UnifiedRecord


def _safe_json_load(path: Path) -> dict | list:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _convert_pheme_annotation(annotation: dict) -> str:
    if annotation.get("is_rumour", "").lower() in {"nonrumour", "non-rumour", "non-rumor"}:
        return "non-rumor"

    misinformation = annotation.get("misinformation")
    truth = annotation.get("true")
    if misinformation is None and truth is None:
        return "unverified"

    if misinformation is not None and truth is None:
        return "false" if int(misinformation) == 1 else "unverified"

    if misinformation is not None and truth is not None:
        misinfo = int(misinformation)
        tr = int(truth)
        if misinfo == 1 and tr == 0:
            return "false"
        if misinfo == 0 and tr == 1:
            return "true"
        if misinfo == 0 and tr == 0:
            return "unverified"

    return "unverified"


def _read_reactions(path: Path) -> list[str]:
    if not path.exists():
        return []
    comments: list[str] = []
    for reaction_file in path.glob("*.json"):
        if reaction_file.name.startswith("._"):
            continue
        reaction = _safe_json_load(reaction_file)
        if not isinstance(reaction, dict):
            continue
        text = clean_text(str(reaction.get("text", "")))
        if text:
            comments.append(text)
    return comments


def load_pheme_dataset(dataset_root: Path, max_records: int | None = None) -> list[UnifiedRecord]:
    records: list[UnifiedRecord] = []
    root = dataset_root / "all-rnr-annotated-threads"
    if not root.exists():
        return records

    for event_dir in root.iterdir():
        if not event_dir.is_dir() or event_dir.name.startswith("."):
            continue

        for rumor_group in ("rumours", "non-rumours"):
            group_dir = event_dir / rumor_group
            if not group_dir.exists():
                continue

            for thread_dir in group_dir.iterdir():
                if not thread_dir.is_dir():
                    continue

                source_dir = thread_dir / "source-tweets"
                source_files = [
                    p for p in source_dir.glob("*.json") if not p.name.startswith("._")
                ] if source_dir.exists() else []
                if not source_files:
                    continue

                source = _safe_json_load(source_files[0])
                if not isinstance(source, dict):
                    continue

                annotation_path = thread_dir / "annotation.json"
                annotation = _safe_json_load(annotation_path) if annotation_path.exists() else {}
                if not isinstance(annotation, dict):
                    annotation = {}

                label = normalize_label(_convert_pheme_annotation(annotation))
                text = clean_text(str(source.get("text", "")))
                thread_id = thread_dir.name
                source_id = str(source.get("id_str") or source.get("id") or thread_id)
                comments = _read_reactions(thread_dir / "reactions")

                metadata = {
                    "event": event_dir.name,
                    "created_at": source.get("created_at"),
                    "user_id": (source.get("user") or {}).get("id_str"),
                    "retweet_count": source.get("retweet_count"),
                    "favorite_count": source.get("favorite_count"),
                    "annotation": annotation,
                }

                records.append(
                    UnifiedRecord(
                        record_id=f"pheme:{event_dir.name}:{thread_id}",
                        dataset="pheme",
                        label=label,
                        text=text,
                        language=source.get("lang"),
                        event=event_dir.name,
                        thread_id=thread_id,
                        source_id=source_id,
                        comments=comments,
                        metadata=metadata,
                    )
                )

                if max_records is not None and len(records) >= max_records:
                    return records

    return records
