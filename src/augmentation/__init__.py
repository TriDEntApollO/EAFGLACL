"""Augmentation interfaces for LLM-driven positive pair generation."""

from .emotion_filter import EmotionFilter, filter_emotion_consistent_candidates

__all__ = ["EmotionFilter", "filter_emotion_consistent_candidates"]
