from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingBatch:
    anchor_texts: list[str]
    positive_texts: list[str]
    labels: list[int]
