import json
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
RESULTS_PATH = BASE_DIRECTORY / "models" / "cross_validation_metrics.json"
LABELS = ("negative", "neutral", "positive")
FOLD_COUNT = 5
RANDOM_STATE = 42

CONFIGURATIONS = (
    {
        "id": "A",
        "name": "TF-IDF unigrams + Logistic Regression",
        "ngram_range": (1, 1),
        "classifier": "logistic_regression",
        "class_weight": None,
    },
    {
        "id": "B",
        "name": "TF-IDF unigrams + bigrams + Logistic Regression",
        "ngram_range": (1, 2),
        "classifier": "logistic_regression",
        "class_weight": None,
    },
    {
        "id": "C",
        "name": "TF-IDF unigrams + bigrams + balanced Logistic Regression",
        "ngram_range": (1, 2),
        "classifier": "logistic_regression",
        "class_weight": "balanced",
    },
    {
        "id": "D",
        "name": "TF-IDF unigrams + bigrams + LinearSVC",
        "ngram_range": (1, 2),
        "classifier": "linear_svc",
        "class_weight": None,
    },
)


def load_training_data() -> tuple[list[str], list[str], dict[str, int]]:
    dataset = pd.read_csv(DATASET_PATH)
    if list(dataset.columns) != ["review", "sentiment"]:
        raise ValueError("Training dataset must contain exactly review,sentiment columns")

    dataset["review"] = dataset["review"].fillna("").astype(str).str.strip()
    dataset["sentiment"] = dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    if dataset["review"].eq("").any():
        raise ValueError("Training dataset contains blank reviews")
    if not dataset["sentiment"].isin(LABELS).all():
        raise ValueError("Training dataset contains invalid sentiment labels")

    class_distribution = {
        label: int((dataset["sentiment"] == label).sum()) for label in LABELS
    }
    if any(count < FOLD_COUNT for count in class_distribution.values()):
        raise ValueError("Each sentiment class must have at least five reviews")

    reviews = dataset["review"].map(normalize_review).tolist()
    labels = dataset["sentiment"].tolist()
    return reviews, labels, class_distribution


def create_classifier(configuration: dict):
    if configuration["classifier"] == "linear_svc":
        return LinearSVC(random_state=RANDOM_STATE)
    return LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=configuration["class_weight"],
    )


def score_fold(actual: list[str], predicted: list[str]) -> dict[str, float]:
    return {
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
        "weighted_f1": float(
            f1_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)
        ),
    }


def summarize_folds(fold_results: list[dict]) -> dict[str, float]:
    accuracy_values = [fold["accuracy"] for fold in fold_results]
    macro_f1_values = [fold["macro_f1"] for fold in fold_results]
    return {
        "mean_accuracy": fmean(accuracy_values),
        "std_accuracy": pstdev(accuracy_values),
        "mean_macro_precision": fmean(fold["macro_precision"] for fold in fold_results),
        "mean_macro_recall": fmean(fold["macro_recall"] for fold in fold_results),
        "mean_macro_f1": fmean(macro_f1_values),
        "std_macro_f1": pstdev(macro_f1_values),
        "mean_weighted_f1": fmean(fold["weighted_f1"] for fold in fold_results),
    }


def print_summary_table(results: dict) -> None:
    headings = (
        ("mean_accuracy", "Mean Acc"),
        ("std_accuracy", "Std Acc"),
        ("mean_macro_precision", "Macro P"),
        ("mean_macro_recall", "Macro R"),
        ("mean_macro_f1", "Mean Macro F1"),
        ("std_macro_f1", "Std Macro F1"),
        ("mean_weighted_f1", "Weighted F1"),
    )
    print("\n5-fold stratified cross-validation (population standard deviation)")
    print(f"{'Model':<5} {'Configuration':<55}" + "".join(f"{title:>15}" for _, title in headings))
    for configuration in CONFIGURATIONS:
        summary = results["models"][configuration["id"]]["summary"]
        row = f"{configuration['id']:<5} {configuration['name']:<55}"
        row += "".join(f"{summary[key]:>15.4f}" for key, _ in headings)
        print(row)


def run_cross_validation() -> dict:
    reviews, labels, class_distribution = load_training_data()
    splitter = StratifiedKFold(
        n_splits=FOLD_COUNT,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    model_results = {}

    for configuration in CONFIGURATIONS:
        fold_results = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(reviews, labels),
            start=1,
        ):
            pipeline = Pipeline(
                steps=(
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=configuration["ngram_range"],
                            min_df=1,
                        ),
                    ),
                    ("classifier", create_classifier(configuration)),
                )
            )
            fold_train_reviews = [reviews[index] for index in train_indices]
            fold_validation_reviews = [reviews[index] for index in validation_indices]
            fold_train_labels = [labels[index] for index in train_indices]
            fold_validation_labels = [labels[index] for index in validation_indices]
            pipeline.fit(fold_train_reviews, fold_train_labels)
            predictions = pipeline.predict(fold_validation_reviews).tolist()
            fold_results.append(
                {
                    "fold": fold_number,
                    "validation_size": len(validation_indices),
                    **score_fold(fold_validation_labels, predictions),
                }
            )

        model_results[configuration["id"]] = {
            "name": configuration["name"],
            "configuration": {
                "tfidf_ngram_range": list(configuration["ngram_range"]),
                "classifier": configuration["classifier"],
                "class_weight": configuration["class_weight"],
                "random_state": RANDOM_STATE,
            },
            "folds": fold_results,
            "summary": summarize_folds(fold_results),
        }

    best_macro_f1 = max(
        CONFIGURATIONS,
        key=lambda configuration: model_results[configuration["id"]]["summary"]["mean_macro_f1"],
    )
    most_consistent = min(
        CONFIGURATIONS,
        key=lambda configuration: model_results[configuration["id"]]["summary"]["std_macro_f1"],
    )
    results = {
        "dataset": {
            "path": "backend/dataset/reviews.csv",
            "size": len(reviews),
            "class_distribution": class_distribution,
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": FOLD_COUNT,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "standard_deviation_ddof": 0,
            "independent_test_set_used": False,
        },
        "models": model_results,
        "highest_mean_macro_f1": {
            "model": best_macro_f1["id"],
            "name": best_macro_f1["name"],
            "mean_macro_f1": model_results[best_macro_f1["id"]]["summary"]["mean_macro_f1"],
        },
        "most_consistent_macro_f1": {
            "model": most_consistent["id"],
            "name": most_consistent["name"],
            "std_macro_f1": model_results[most_consistent["id"]]["summary"]["std_macro_f1"],
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_summary_table(results)
    print(f"\nHighest mean macro F1: {results['highest_mean_macro_f1']}")
    print(f"Most consistent macro F1: {results['most_consistent_macro_f1']}")
    print(f"Saved cross-validation results: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_cross_validation()