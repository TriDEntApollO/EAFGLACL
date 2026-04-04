"""Emotion analysis utilities used by the emotion enrichment phase."""

from .filter import EmotionFilter, filter_emotion_consistent_candidates

__all__ = [
    "EmotionFilter",
    "filter_emotion_consistent_candidates",
]