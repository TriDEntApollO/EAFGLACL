from __future__ import annotations

import html
import re
from typing import Final

URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")

LABEL_MAP: Final[dict[str, str]] = {
    "fake": "false",
    "false": "false",
    "true": "true",
    "real": "true",
    "unverified": "unverified",
    "non-rumor": "non-rumor",
    "non-rumour": "non-rumor",
    "nonrumor": "non-rumor",
    "nonrumour": "non-rumor",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]
    raise ValueError(f"Unsupported label: {label!r}")


def strip_html(text: str) -> str:
    if "<" not in text and "&" not in text:
        return text
    unescaped = html.unescape(text)
    without_tags = HTML_TAG_PATTERN.sub(" ", unescaped)
    return WHITESPACE_PATTERN.sub(" ", without_tags).strip()


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = text.replace("URL", "")
    text = URL_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
