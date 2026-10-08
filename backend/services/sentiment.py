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
            f"Sentiment model files are unavailable. "
            f"Run `python train_sentiment.py` from the backend directory. "
            f"Details: {error}"
        )


def analyze_sentiment(review: str) -> dict:
    if _vectorizer is None or _classifier is None:
        raise RuntimeError(
            _load_error or "The trained sentiment model is not loaded."
        )

    normalized_review = normalize_review(review)
    features = _vectorizer.transform([normalized_review])

    sentiment = str(_classifier.predict(features)[0]).lower()

    # Logistic Regression provides class probabilities.
    if hasattr(_classifier, "predict_proba"):
        probabilities = _classifier.predict_proba(features)[0]

        classes = [str(label).lower() for label in _classifier.classes_]

        predicted_index = classes.index(sentiment)

        confidence = float(probabilities[predicted_index]) * 100.0
        confidence = round(confidence, 2)
    else:
        confidence = None

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "explanation": _EXPLANATIONS[sentiment],
    }