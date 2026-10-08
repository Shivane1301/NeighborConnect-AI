import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
TRAINING_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
TEST_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_test.csv"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
VECTORIZER_PATH = MODEL_DIRECTORY / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIRECTORY / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIRECTORY / "evaluation_metrics.json"
LABELS = ("negative", "neutral", "positive")


def read_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if list(dataset.columns) != ["review", "sentiment"]:
        raise ValueError(f"{path.name} must contain exactly review,sentiment columns")

    dataset["review"] = dataset["review"].fillna("").astype(str).str.strip()
    dataset["sentiment"] = dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    if dataset["review"].eq("").any():
        raise ValueError(f"{path.name} contains empty reviews")
    if not dataset["sentiment"].isin(LABELS).all():
        raise ValueError(f"{path.name} contains invalid sentiment labels")
    return dataset


def evaluate() -> dict:
    training_dataset = read_dataset(TRAINING_DATASET_PATH)
    test_dataset = read_dataset(TEST_DATASET_PATH)

    normalized_training_reviews = {
        normalize_review(review) for review in training_dataset["review"]
    }
    normalized_test_reviews = [
        normalize_review(review) for review in test_dataset["review"]
    ]
    duplicate_count = len(normalized_test_reviews) - len(set(normalized_test_reviews))
    overlap_count = len(
        normalized_training_reviews.intersection(normalized_test_reviews)
    )
    if duplicate_count:
        raise ValueError(f"Test dataset contains {duplicate_count} duplicate reviews")
    if overlap_count:
        raise ValueError(
            f"Test dataset overlaps the training dataset by {overlap_count} reviews"
        )
    if len(test_dataset) != 150:
        raise ValueError(f"Expected 150 test reviews, found {len(test_dataset)}")

    class_distribution = {
        label: int((test_dataset["sentiment"] == label).sum()) for label in LABELS
    }
    if class_distribution != {label: 50 for label in LABELS}:
        raise ValueError(f"Test dataset must have exactly 50 reviews per class: {class_distribution}")

    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    features = vectorizer.transform(normalized_test_reviews)
    actual = test_dataset["sentiment"].tolist()
    predicted = [str(label).lower() for label in classifier.predict(features)]

    report = classification_report(
        actual,
        predicted,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "training_dataset_size": int(len(training_dataset)),
        "test_dataset_size": int(len(test_dataset)),
        "class_distribution": class_distribution,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)
        ),
        "per_class_metrics": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "classification_report": report,
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": confusion_matrix(actual, predicted, labels=list(LABELS)).tolist(),
        },
        "duplicate_review_count": duplicate_count,
        "training_test_overlap_count": overlap_count,
        "model_loaded": True,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()