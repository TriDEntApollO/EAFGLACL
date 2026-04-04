from __future__ import annotations

from data_prep.schema import UnifiedRecord
from emotion.filter import EmotionFilter


def enrich_records(
    records: list[UnifiedRecord],
    enable_emotion: bool = False,
    enable_fact: bool = False,
) -> list[UnifiedRecord]:
    if not records:
        return records

    emotion_model = EmotionFilter() if enable_emotion else None
    fact_verifier = None
    if enable_fact:
        from fact.wikipedia_verifier import WikipediaFactVerifier

        fact_verifier = WikipediaFactVerifier()

    for record in records:
        if emotion_model is not None:
            emotion, confidence = emotion_model.predict_emotion(record.text)
            record.emotion = emotion
            record.emotion_confidence = confidence

        if fact_verifier is not None:
            record.fact_consistency_score = fact_verifier.fact_consistency_score(record.text)

    return records
