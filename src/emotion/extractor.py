import torch
import warnings  # 🔥 Import Python's built-in warnings
from transformers import pipeline, logging
from .config import STANDARD_EMOTIONS

# Silence Hugging Face's custom logger
logging.set_verbosity_error()

# 🔥 Silence Python's built-in UserWarnings coming from the pipeline
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class EmotionExtractor:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1

        if device == 0:
            print(f"⚡ GPU Detected! Running inference on: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU detected by PyTorch. Running on CPU...")

        self.model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device,
            batch_size=16
        )

    def get_emotion_distribution(self, text: str) -> dict:
        if not text:
            return {e: 0.0 for e in STANDARD_EMOTIONS}

        raw_scores = self.model(text)[0]
        distribution = {item["label"]: item["score"] for item in raw_scores}

        return {e: distribution.get(e, 0.0) for e in STANDARD_EMOTIONS}

    def get_emotion_distributions_batch(self, texts: list) -> list:
        if not texts:
            return []

        raw_scores_batch = self.model(texts)

        results = []
        for raw_scores in raw_scores_batch:
            distribution = {item["label"]: item["score"] for item in raw_scores}
            results.append({e: distribution.get(e, 0.0) for e in STANDARD_EMOTIONS})

        return results