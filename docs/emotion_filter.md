# Emotion-Preserving Filter for LACL

This module is a core enhancement to the LLM-Augmented Contrastive Learning (LACL) framework for misinformation detection. It ensures that LLM-generated data augmentations preserve the original emotional tone (e.g., anger, fear, sarcasm) of the source text, preventing the creation of false-positive pairs in the contrastive learning latent space.

---

## 🧠 Research Context

In Contrastive Learning, the quality of positive pairs dictates the encoder's performance. If an LLM augments a piece of highly fearful misinformation into a polite, neutral summary, forcing those embeddings together will degrade the model.

This module extracts a multi-dimensional emotion vector for both:
- the anchor (original text)  
- the candidate (augmented text)  

It then uses dynamic cosine similarity thresholding to discard emotionally inconsistent augmentations, ensuring only high-quality, emotion-consistent pairs are used during training.

---

## ⚙️ Architecture

The module is designed to be modular, scalable, and efficient, and can be integrated into larger pipelines.

### 🔹 Core Components

- config.py
  - Defines:
    - Standard emotion labels
    - Threshold values
  - Acts as the central configuration hub

- extractor.py
  - Handles emotion extraction using:
    j-hartmann/emotion-english-distilroberta-base
  - Outputs emotion distributions (e.g., anger, fear, joy, etc.)
  - Supports batch processing (optional optimization)

- filter.py
  - Core logic of the system
  - Performs:
    - Emotion vector conversion
    - Cosine similarity calculation
    - Threshold-based filtering

- batch_processor.py (optional but recommended)
  - Handles dataset-level processing
  - Groups augmentations per original sample
  - Avoids redundant emotion computation

- main.py
  - Entry point of the module
  - Runs full pipeline:
    Load → Filter → Merge → Output

---

## 🧮 Dynamic Thresholding

A fixed threshold is not sufficient for misinformation detection because emotional intensity varies significantly.

### 🔥 Strategy:

- High Emotional Intensity
  - Threshold becomes stricter
- Low / Neutral Emotion
  - Threshold is relaxed

Example:
If max_emotion_score > 0.85 → threshold ≈ 0.90–0.95  
If max_emotion_score < 0.40 → threshold ≈ 0.50–0.70

---

## 📂 Data Flow

### Input Format

{
  "source_record_id": "twitter15:123",
  "source_text": "Original misinformation text...",
  "candidate_text": "LLM generated augmentation..."
}

### Processing Steps

Original Text → Emotion Extraction → Augmented Text → Emotion Extraction → Similarity → Filter

### Output Format

{
  "text": "Filtered augmented text",
  "emotion_similarity": 0.91
}

---

## 🚀 Usage

Activate environment and run:

python src/emotion/main.py

---

## 🎯 Key Contribution

✔ Preserves emotional consistency  
✔ Reduces noisy augmentations  
✔ Improves contrastive learning  
✔ Enhances misinformation detection  

---

## 🧠 Intuition

LLM → Generates  
Emotion Filter → Validates  
Model → Learns  

---

## 📌 Summary

“Positive pairs are not just semantically similar, but also emotionally aligned.”

---

## 🔜 Next Step

Integrate Fact Verification Module for truth-awareness.
