import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
TRAINING_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
INDEPENDENT_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_test.csv"
RESULTS_PATH = BASE_DIRECTORY / "models" / "model_comparison.json"
LABELS = ("negative", "neutral", "positive")
RANDOM_STATE = 42
INTERNAL_TEST_SIZE = 0.25

CONFIGURATIONS = (
    {
        "id": "A",
        "name": "Unigrams + Logistic Regression",
        "ngram_range": (1, 1),
        "classifier": "logistic_regression",
        "class_weight": None,
    },
    {
        "id": "B",
        "name": "Unigrams + bigrams + Logistic Regression",
        "ngram_range": (1, 2),
        "classifier": "logistic_regression",
        "class_weight": None,
    },
    {
        "id": "C",
        "name": "Unigrams + bigrams + balanced Logistic Regression",
        "ngram_range": (1, 2),
        "classifier": "logistic_regression",
        "class_weight": "balanced",
    },
    {
        "id": "D",
        "name": "Unigrams + bigrams + LinearSVC",
        "ngram_range": (1, 2),
        "classifier": "linear_svc",
        "class_weight": None,
    },
)


def read_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if list(dataset.columns) != ["review", "sentiment"]:
        raise ValueError(f"{path.name} must contain exactly review,sentiment columns")
    dataset["review"] = dataset["review"].fillna("").astype(str).str.strip()
    dataset["sentiment"] = dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    if dataset["review"].eq("").any():
        raise ValueError(f"{path.name} contains blank reviews")
    if not dataset["sentiment"].isin(LABELS).all():
        raise ValueError(f"{path.name} contains invalid sentiment labels")
    return dataset


def calculate_metrics(actual: list[str], predicted: list[str]) -> dict:
    report = classification_report(
        actual,
        predicted,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
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
    }
    return metrics


def create_classifier(configuration: dict):
    if configuration["classifier"] == "linear_svc":
        return LinearSVC(random_state=RANDOM_STATE)
    return LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=configuration["class_weight"],
    )


def print_comparison_table(title: str, model_results: dict, result_key: str) -> None:
    columns = (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro P"),
        ("macro_recall", "Macro R"),
        ("macro_f1", "Macro F1"),
        ("weighted_precision", "Weighted P"),
        ("weighted_recall", "Weighted R"),
        ("weighted_f1", "Weighted F1"),
    )
    print(f"\n{title}")
    print(f"{'Model':<5}" + "".join(f"{heading:>13}" for _, heading in columns))
    for configuration in CONFIGURATIONS:
        values = model_results[configuration["id"]][result_key]
        row = f"{configuration['id']:<5}" + "".join(
            f"{values[key]:>13.4f}" for key, _ in columns
        )
        print(row)


def run_comparison() -> dict:
    training_dataset = read_dataset(TRAINING_DATASET_PATH)
    independent_dataset = read_dataset(INDEPENDENT_DATASET_PATH)
    independent_counts = Counter(independent_dataset["sentiment"])
    expected_counts = {label: 50 for label in LABELS}
    if len(independent_dataset) != 150 or dict(independent_counts) != expected_counts:
        raise ValueError(
            "Independent test set must contain exactly 150 reviews, 50 per class; "
            f"found {len(independent_dataset)} and {dict(independent_counts)}"
        )

    training_reviews = training_dataset["review"].map(normalize_review)
    training_labels = training_dataset["sentiment"]
    validation_train_reviews, validation_reviews, validation_train_labels, validation_labels = (
        train_test_split(
            training_reviews,
            training_labels,
            test_size=INTERNAL_TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=training_labels,
        )
    )
    independent_reviews = independent_dataset["review"].map(normalize_review)
    independent_labels = independent_dataset["sentiment"].tolist()

    model_results = {}
    for configuration in CONFIGURATIONS:
        vectorizer = TfidfVectorizer(
            ngram_range=configuration["ngram_range"],
            min_df=1,
        )
        training_features = vectorizer.fit_transform(validation_train_reviews)
        validation_features = vectorizer.transform(validation_reviews)
        independent_features = vectorizer.transform(independent_reviews)

        classifier = create_classifier(configuration)
        classifier.fit(training_features, validation_train_labels)

        validation_predictions = classifier.predict(validation_features).tolist()
        independent_predictions = classifier.predict(independent_features).tolist()
        independent_metrics = calculate_metrics(independent_labels, independent_predictions)
        independent_metrics["confusion_matrix"] = {
            "labels": list(LABELS),
            "matrix": confusion_matrix(
                independent_labels,
                independent_predictions,
                labels=list(LABELS),
            ).tolist(),
        }

        model_results[configuration["id"]] = {
            "name": configuration["name"],
            "configuration": {
                "tfidf_ngram_range": list(configuration["ngram_range"]),
                "classifier": configuration["classifier"],
                "class_weight": configuration["class_weight"],
                "random_state": RANDOM_STATE,
            },
            "internal_validation": calculate_metrics(
                validation_labels.tolist(), validation_predictions
            ),
            "independent_test": independent_metrics,
        }

    best_configuration = max(
        CONFIGURATIONS,
        key=lambda configuration: model_results[configuration["id"]]["independent_test"]["macro_f1"],
    )
    results = {
        "training_dataset_size": int(len(training_dataset)),
        "training_class_distribution": {
            label: int((training_dataset["sentiment"] == label).sum())
            for label in LABELS
        },
        "internal_split": {
            "random_state": RANDOM_STATE,
            "test_size_fraction": INTERNAL_TEST_SIZE,
            "training_samples": int(len(validation_train_reviews)),
            "validation_samples": int(len(validation_reviews)),
        },
        "independent_test": {
            "dataset_size": int(len(independent_dataset)),
            "class_distribution": {
                label: int((independent_dataset["sentiment"] == label).sum())
                for label in LABELS
            },
            "used_for_training_or_tuning": False,
        },
        "models": model_results,
        "best_independent_macro_f1": {
            "model": best_configuration["id"],
            "name": best_configuration["name"],
            "macro_f1": model_results[best_configuration["id"]]["independent_test"]["macro_f1"],
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_comparison_table("Internal validation (shared stratified split)", model_results, "internal_validation")
    print_comparison_table("Independent test set", model_results, "independent_test")
    print("\nIndependent-test confusion matrices (labels: negative, neutral, positive)")
    for configuration in CONFIGURATIONS:
        confusion = model_results[configuration["id"]]["independent_test"]["confusion_matrix"]["matrix"]
        print(f"Model {configuration['id']}: {confusion}")
    print(
        "\nBest independent macro F1: "
        f"Model {best_configuration['id']} ({best_configuration['name']}), "
        f"{results['best_independent_macro_f1']['macro_f1']:.4f}"
    )
    print(f"\nSaved comparison JSON: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_comparison()