import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from .config import STANDARD_EMOTIONS, BASE_SIMILARITY_THRESHOLD


def emotion_to_vector(emotion_dict: dict) -> np.ndarray:
    """Converts a standardized emotion dict into a 7D numpy array."""
    return np.array([emotion_dict.get(e, 0.0) for e in STANDARD_EMOTIONS])


def compute_emotion_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity safely."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(cosine_similarity([vec1], [vec2])[0][0])


def calculate_dynamic_threshold(orig_dist: dict, base_thresh: float) -> float:
    """Tightens threshold for extreme emotions, relaxes for ambiguous ones."""
    max_score = max(orig_dist.values()) if orig_dist else 0.0
    if max_score > 0.85:
        return min(0.95, base_thresh + 0.1)
    elif max_score < 0.4:
        return max(0.50, base_thresh - 0.15)
    return base_thresh


def filter_augmentations(
        original_dist: dict,
        augmented_records: List[Dict],
        augmented_dists: List[dict]
) -> List[Dict]:
    """Filters the dictionary records based on dynamic cosine similarity."""
    dynamic_thresh = calculate_dynamic_threshold(original_dist, BASE_SIMILARITY_THRESHOLD)
    v_orig = emotion_to_vector(original_dist)

    results = []
    for aug_record, aug_dist in zip(augmented_records, augmented_dists):
        v_aug = emotion_to_vector(aug_dist)
        similarity = compute_emotion_similarity(v_orig, v_aug)

        if similarity >= dynamic_thresh:
            enriched_aug = aug_record.copy()
            # Inject our new metadata so we can analyze it later
            enriched_aug["emotion_similarity"] = similarity
            results.append(enriched_aug)

    # Return sorted by best emotional match
    return sorted(results, key=lambda x: x["emotion_similarity"], reverse=True)