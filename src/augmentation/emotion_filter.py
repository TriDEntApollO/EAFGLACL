from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class EmotionFilter:
    """
    Lightweight placeholder emotion filter.

    Replace `predict_emotion` with a transformer-based classifier for production use.
    """

    threshold: float = 0.5

    def predict_emotion(self, text: str) -> tuple[str, float]:
        lowered = text.lower()
        if any(token in lowered for token in ("angry", "rage", "furious")):
            return ("anger", 0.8)
        if any(token in lowered for token in ("fear", "scared", "panic")):
            return ("fear", 0.8)
        if any(token in lowered for token in ("lol", "haha", "sarcasm", "/s")):
            return ("sarcasm", 0.7)
        return ("neutral", 0.6)


def filter_emotion_consistent_candidates(
    source_text: str, candidates: Iterable[str], classifier: EmotionFilter
) -> list[str]:
    src_emotion, src_score = classifier.predict_emotion(source_text)
    if src_score < classifier.threshold:
        return list(candidates)

    kept: list[str] = []
    for candidate in candidates:
        cand_emotion, cand_score = classifier.predict_emotion(candidate)
        if cand_emotion == src_emotion and cand_score >= classifier.threshold:
            kept.append(candidate)
    return kept
