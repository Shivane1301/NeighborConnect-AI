import re
from pathlib import Path

import joblib

MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models"
VECTORIZER_PATH = MODEL_DIRECTORY / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIRECTORY / "sentiment_model.joblib"

_vectorizer = None
_classifier = None
_load_error = None


_EXPLANATIONS = {
    "positive": "The review contains language associated with satisfaction or successful service.",
    "negative": "The review contains language associated with dissatisfaction or service problems.",
    "neutral": "The review is mostly factual or does not strongly express positive or negative sentiment.",
}


def normalize_review(review: str) -> str:
    """Apply the same lightweight normalization during training and inference."""
    return re.sub(r"\s+", " ", review.lower()).strip()


def load_sentiment_model() -> None:
    """Load trained artifacts once when the FastAPI application starts."""
    global _vectorizer, _classifier, _load_error
    try:
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
        _load_error = None
    except (FileNotFoundError, OSError, ValueError) as error:
        _vectorizer = None
        _classifier = None
        _load_error = (
            f"Sentiment model files are unavailable. Run `python train_sentiment.py` "
            f"from the backend directory. Details: {error}"
        )


def analyze_sentiment(review: str) -> dict:
    if _vectorizer is None or _classifier is None:
        raise RuntimeError(_load_error or "The trained sentiment model is not loaded.")

    features = _vectorizer.transform([normalize_review(review)])
    probabilities = _classifier.predict_proba(features)[0]
    labels = _classifier.classes_
    best_index = probabilities.argmax()
    sentiment = str(labels[best_index]).lower()
    return {
        "sentiment": sentiment,
        "confidence": round(float(probabilities[best_index]), 4),
        "explanation": _EXPLANATIONS[sentiment],
    }
