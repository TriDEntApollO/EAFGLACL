from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class UnifiedExample:
    example_id: str
    dataset: str
    split: str
    label: str
    text: str
    source_id: str
    language: str = "unknown"
    emotion: Optional[str] = None
    fact_consistency_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = data["metadata"] or {}
        return data


@dataclass
class DatasetSummary:
    dataset: str
    total_examples: int
    label_counts: Dict[str, int]
    notes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["notes"] = data["notes"] or []
        return data
