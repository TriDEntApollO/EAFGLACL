from __future__ import annotations

from augmentation.emotion_filter import EmotionFilter
from data.schema import UnifiedRecord
from fact.wikipedia_verifier import WikipediaFactVerifier


def enrich_records(
    records: list[UnifiedRecord],
    enable_emotion: bool = False,
    enable_fact: bool = False,
) -> list[UnifiedRecord]:
    if not records:
        return records

    emotion_model = EmotionFilter() if enable_emotion else None
    fact_verifier = WikipediaFactVerifier() if enable_fact else None

    for record in records:
        if emotion_model is not None:
            emotion, confidence = emotion_model.predict_emotion(record.text)
            record.emotion = emotion
            record.emotion_confidence = confidence

        if fact_verifier is not None:
            record.fact_consistency_score = fact_verifier.fact_consistency_score(record.text)

    return records
