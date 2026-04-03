from __future__ import annotations

import json
from pathlib import Path

from .normalization import clean_text
from .schema import UnifiedRecord


def _safe_json_load(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _extract_post_text(post: dict) -> str:
    raw = post.get("text_raw") or post.get("text") or ""
    return clean_text(str(raw))


def _extract_comments(comment_payload: list[dict]) -> list[str]:
    comments: list[str] = []
    for item in comment_payload:
        raw = item.get("text_raw") or item.get("text") or ""
        text = clean_text(str(raw))
        if text:
            comments.append(text)
    return comments


def load_weibo_dataset(dataset_root: Path, max_records: int | None = None) -> list[UnifiedRecord]:
    records: list[UnifiedRecord] = []
    label_dirs = [("fake_news", "false"), ("real_news", "true")]

    for subdir_name, label in label_dirs:
        partition_root = dataset_root / subdir_name
        if not partition_root.exists():
            continue

        for item_dir in partition_root.iterdir():
            if not item_dir.is_dir():
                continue

            post_path = item_dir / "new.json"
            comments_path = item_dir / "comments.json"
            if not post_path.exists():
                continue

            post_json = _safe_json_load(post_path)
            comments_json = _safe_json_load(comments_path) if comments_path.exists() else []
            if not isinstance(post_json, dict):
                continue
            if not isinstance(comments_json, list):
                comments_json = []

            source_id = str(post_json.get("idstr") or post_json.get("id") or item_dir.name)
            text = _extract_post_text(post_json)
            comments = _extract_comments(comments_json)

            metadata = {
                "mblogid": post_json.get("mblogid"),
                "user_id": (post_json.get("user") or {}).get("idstr"),
                "created_at": post_json.get("created_at"),
                "reposts_count": post_json.get("reposts_count"),
                "comments_count": post_json.get("comments_count"),
                "attitudes_count": post_json.get("attitudes_count"),
            }

            records.append(
                UnifiedRecord(
                    record_id=f"weibo:{item_dir.name}",
                    dataset="weibo",
                    label=label,
                    text=text,
                    language="zh",
                    thread_id=item_dir.name,
                    source_id=source_id,
                    comments=comments,
                    metadata=metadata,
                )
            )

            if max_records is not None and len(records) >= max_records:
                return records

    return records
