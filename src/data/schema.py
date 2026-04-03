from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UnifiedRecord:
    record_id: str
    dataset: str
    label: str
    text: str
    language: str | None = None
    event: str | None = None
    thread_id: str | None = None
    source_id: str | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None
    fact_consistency_score: float | None = None
    comments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
