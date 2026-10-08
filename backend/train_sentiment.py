import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from services.sentiment import normalize_review

BASE_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
VECTORIZER_PATH = MODEL_DIRECTORY / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIRECTORY / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIRECTORY / "training_metrics.json"
LABELS = ("negative", "neutral", "positive")
INFERENCE_CASES = (
    ("The plumber arrived quickly and fixed the leaking tap perfectly.", "positive"),
    ("The electrician arrived late and did not complete the repair.", "negative"),
    ("The repair was completed at the scheduled time.", "neutral"),
)


def load_dataset() -> tuple[pd.DataFrame, dict]:
    dataset = pd.read_csv(DATASET_PATH)
    if list(dataset.columns) != ["review", "sentiment"]:
        raise ValueError("Dataset must contain exactly review,sentiment columns")

    dataset["review"] = dataset["review"].fillna("").astype(str).str.strip()
    dataset["sentiment"] = dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    empty_review_count = int(dataset["review"].eq("").sum())
    invalid_label_count = int((~dataset["sentiment"].isin(LABELS)).sum())
    dataset = dataset[
        dataset["review"].ne("") & dataset["sentiment"].isin(LABELS)
    ].copy()
    duplicate_count = int(dataset["review"].duplicated().sum())
    dataset = dataset.drop_duplicates(subset="review").copy()

    if dataset.empty:
        raise ValueError("Dataset has no valid labeled reviews")

    class_distribution = {
        label: int(dataset["sentiment"].eq(label).sum()) for label in LABELS
    }
    if any(count < 2 for count in class_distribution.values()):
        raise ValueError("Each sentiment class needs at least two valid reviews")

    quality = {
        "dataset_size": int(len(dataset)),
        "class_distribution": class_distribution,
        "duplicate_count": duplicate_count,
        "empty_review_count": empty_review_count,
        "invalid_label_count": invalid_label_count,
        "minimum_review_length": int(dataset["review"].str.len().min()),
        "maximum_review_length": int(dataset["review"].str.len().max()),
    }
    dataset["review"] = dataset["review"].map(normalize_review)
    return dataset, quality


def verify_inference(vectorizer_path: Path, classifier_path: Path) -> list[dict]:
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    results = []
    for review, expected in INFERENCE_CASES:
        features = vectorizer.transform([normalize_review(review)])
        predicted = str(classifier.predict(features)[0]).lower()
        if predicted not in LABELS:
            raise RuntimeError(f"Unexpected sentiment label from saved model: {predicted}")
        results.append({"expected": expected, "predicted": predicted})
    return results


def train() -> dict:
    dataset, quality = load_dataset()

    training_reviews, testing_reviews, training_labels, testing_labels = train_test_split(
        dataset["review"],
        dataset["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=dataset["sentiment"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    training_features = vectorizer.fit_transform(training_reviews)
    testing_features = vectorizer.transform(testing_reviews)

    classifier = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    classifier.fit(training_features, training_labels)
    predictions = classifier.predict(testing_features)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        testing_labels,
        predictions,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    macro_average = {
        "precision": float(precision_score(testing_labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(testing_labels, predictions, average="macro", zero_division=0)),
        "f1_score": float(f1_score(testing_labels, predictions, average="macro", zero_division=0)),
    }
    weighted_average = {
        "precision": float(precision_score(testing_labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(testing_labels, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(testing_labels, predictions, average="weighted", zero_division=0)),
    }
    evaluation = {
        **quality,
        "train_size": int(len(training_reviews)),
        "test_size": int(len(testing_reviews)),
        "accuracy": float(accuracy_score(testing_labels, predictions)),
        "precision": macro_average["precision"],
        "recall": macro_average["recall"],
        "f1_score": macro_average["f1_score"],
        "macro_average": macro_average,
        "weighted_average": weighted_average,
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
            "matrix": confusion_matrix(
                testing_labels, predictions, labels=list(LABELS)
            ).tolist(),
        },
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory(prefix=".sentiment-training-", dir=MODEL_DIRECTORY) as temp_dir:
        temp_dir = Path(temp_dir)
        staged_vectorizer = temp_dir / VECTORIZER_PATH.name
        staged_classifier = temp_dir / CLASSIFIER_PATH.name
        staged_metrics = temp_dir / METRICS_PATH.name
        joblib.dump(vectorizer, staged_vectorizer)
        joblib.dump(classifier, staged_classifier)
        staged_metrics.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

        staged_inference = verify_inference(staged_vectorizer, staged_classifier)
        os.replace(staged_vectorizer, VECTORIZER_PATH)
        os.replace(staged_classifier, CLASSIFIER_PATH)
        os.replace(staged_metrics, METRICS_PATH)

    saved_inference = verify_inference(VECTORIZER_PATH, CLASSIFIER_PATH)
    evaluation["inference_tests"] = [
        {
            "review": review,
            "expected": expected,
            "predicted": result["predicted"],
            "matches_expected": result["predicted"] == expected,
        }
        for (review, expected), result in zip(INFERENCE_CASES, saved_inference)
    ]
    METRICS_PATH.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset samples: {quality['dataset_size']}")
    print(f"Class distribution: {quality['class_distribution']}")
    print(f"Duplicate count: {quality['duplicate_count']}")
    print(f"Empty review count: {quality['empty_review_count']}")
    print(f"Minimum review length: {quality['minimum_review_length']}")
    print(f"Maximum review length: {quality['maximum_review_length']}")
    print(f"Training samples: {len(training_reviews)}")
    print(f"Testing samples: {len(testing_reviews)}")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro precision: {macro_average['precision']:.4f}")
    print(f"Macro recall: {macro_average['recall']:.4f}")
    print(f"Macro F1-score: {macro_average['f1_score']:.4f}")
    print(f"Weighted metrics: {weighted_average}")
    print("Classification report:")
    print(classification_report(testing_labels, predictions, labels=list(LABELS), zero_division=0))
    print(f"Confusion matrix (labels {list(LABELS)}):")
    print(evaluation["confusion_matrix"]["matrix"])
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved classifier: {CLASSIFIER_PATH}")
    print(f"Staged inference checks: {staged_inference}")
    print(f"Saved artifact inference checks: {saved_inference}")
    return evaluation


if __name__ == "__main__":
    train()
