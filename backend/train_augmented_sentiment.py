import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_diverse_augmented.csv"
TARGETED_PATH = BASE_DIRECTORY / "dataset" / "reviews_targeted.csv"
TEST_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_test.csv"
PRODUCTION_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
DIVERSE_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_diverse.csv"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
RESULTS_PATH = MODEL_DIRECTORY / "augmented_model_comparison.json"
OLD_RESULTS_PATH = MODEL_DIRECTORY / "diverse_model_comparison.json"
LABELS = ("negative", "neutral", "positive")
FOLD_COUNT = 5
RANDOM_STATE = 42

CONFIGURATIONS = (
    {
        "id": "A",
        "name": "TF-IDF unigrams + Logistic Regression",
        "ngram_range": (1, 1),
        "classifier": "logistic_regression",
    },
    {
        "id": "D",
        "name": "TF-IDF unigrams + bigrams + LinearSVC",
        "ngram_range": (1, 2),
        "classifier": "linear_svc",
    },
)

PROTECTED_PRODUCTION_PATHS = {
    "original_training_dataset": PRODUCTION_DATASET_PATH,
    "diverse_training_dataset": DIVERSE_DATASET_PATH,
    "targeted_dataset": TARGETED_PATH,
    "augmented_training_dataset": DATASET_PATH,
    "production_classifier": MODEL_DIRECTORY / "sentiment_model.joblib",
    "production_vectorizer": MODEL_DIRECTORY / "tfidf_vectorizer.joblib",
}


def artifact_paths(model_id: str) -> tuple[Path, Path]:
    suffix = model_id.lower()
    return (
        MODEL_DIRECTORY / f"experimental_augmented_vectorizer_{suffix}.joblib",
        MODEL_DIRECTORY / f"experimental_augmented_{suffix}.joblib",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def create_classifier(configuration: dict):
    if configuration["classifier"] == "linear_svc":
        return LinearSVC(random_state=RANDOM_STATE)
    return LogisticRegression(C=5.0, max_iter=1000, random_state=RANDOM_STATE)


def calculate_metrics(actual: list[str], predicted: list[str]) -> dict:
    report = classification_report(
        actual,
        predicted,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
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
        "per_class_metrics": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def cross_validate(reviews: list[str], labels: list[str]) -> dict:
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for configuration in CONFIGURATIONS:
        folds = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(reviews, labels),
            start=1,
        ):
            pipeline = Pipeline(
                steps=(
                    (
                        "tfidf",
                        TfidfVectorizer(ngram_range=configuration["ngram_range"], min_df=1),
                    ),
                    ("classifier", create_classifier(configuration)),
                )
            )
            train_reviews = [reviews[index] for index in train_indices]
            validation_reviews = [reviews[index] for index in validation_indices]
            train_labels = [labels[index] for index in train_indices]
            validation_labels = [labels[index] for index in validation_indices]
            pipeline.fit(train_reviews, train_labels)
            predictions = pipeline.predict(validation_reviews).tolist()
            folds.append(
                {
                    "fold": fold_number,
                    "validation_size": len(validation_indices),
                    **calculate_metrics(validation_labels, predictions),
                }
            )

        results[configuration["id"]] = {
            "mean_accuracy": fmean(fold["accuracy"] for fold in folds),
            "std_accuracy": pstdev(fold["accuracy"] for fold in folds),
            "mean_macro_precision": fmean(fold["macro_precision"] for fold in folds),
            "mean_macro_recall": fmean(fold["macro_recall"] for fold in folds),
            "mean_macro_f1": fmean(fold["macro_f1"] for fold in folds),
            "std_macro_f1": pstdev(fold["macro_f1"] for fold in folds),
            "mean_weighted_f1": fmean(fold["weighted_f1"] for fold in folds),
            "folds": folds,
        }
    return results


def print_cv_table(results: dict) -> None:
    print("\n5-fold StratifiedKFold on augmented training data")
    print("Model | Mean accuracy +/- SD | Mean macro F1 +/- SD | Mean weighted F1")
    for configuration in CONFIGURATIONS:
        summary = results[configuration["id"]]
        print(
            f"{configuration['id']} | {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f} | "
            f"{summary['mean_macro_f1']:.4f} +/- {summary['std_macro_f1']:.4f} | "
            f"{summary['mean_weighted_f1']:.4f}"
        )


def print_test_table(results: dict) -> None:
    print("\nIndependent locked test results")
    print("Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1")
    for configuration in CONFIGURATIONS:
        metrics = results[configuration["id"]]["independent_test"]
        print(
            f"{configuration['id']} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | "
            f"{metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f}"
        )
        print(f"  Confusion matrix [negative, neutral, positive]: {metrics['confusion_matrix']['matrix']}")


def main() -> dict:
    output_paths = [RESULTS_PATH]
    for configuration in CONFIGURATIONS:
        output_paths.extend(artifact_paths(configuration["id"]))
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing experiment outputs: {existing}")

    protected_hashes_before = {
        name: sha256(path) for name, path in PROTECTED_PRODUCTION_PATHS.items()
    }
    training_dataset = read_dataset(DATASET_PATH)
    labels = training_dataset["sentiment"].tolist()
    distribution = Counter(labels)
    expected_distribution = {label: 600 for label in LABELS}
    if len(training_dataset) != 1800 or dict(distribution) != expected_distribution:
        raise ValueError(
            f"Augmented dataset must contain 1,800 reviews, 600 per class; got {len(training_dataset)}, {dict(distribution)}"
        )
    reviews = training_dataset["review"].map(normalize_review).tolist()

    cv_results = cross_validate(reviews, labels)

    # The locked test set is first opened after every CV fold has finished.
    test_hash_before = sha256(TEST_DATASET_PATH)
    test_dataset = read_dataset(TEST_DATASET_PATH)
    test_distribution = Counter(test_dataset["sentiment"])
    expected_test_distribution = {label: 50 for label in LABELS}
    if len(test_dataset) != 150 or dict(test_distribution) != expected_test_distribution:
        raise ValueError(
            f"Locked test set must contain 150 reviews, 50 per class; got {len(test_dataset)}, {dict(test_distribution)}"
        )
    test_reviews = test_dataset["review"].map(normalize_review).tolist()
    test_labels = test_dataset["sentiment"].tolist()

    model_results = {}
    for configuration in CONFIGURATIONS:
        vectorizer = TfidfVectorizer(ngram_range=configuration["ngram_range"], min_df=1)
        train_features = vectorizer.fit_transform(reviews)
        classifier = create_classifier(configuration)
        classifier.fit(train_features, labels)

        vectorizer_path, classifier_path = artifact_paths(configuration["id"])
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)

        predictions = classifier.predict(vectorizer.transform(test_reviews)).tolist()
        independent_metrics = calculate_metrics(test_labels, predictions)
        independent_metrics["weighted_precision"] = float(
            precision_score(test_labels, predictions, labels=list(LABELS), average="weighted", zero_division=0)
        )
        independent_metrics["weighted_recall"] = float(
            recall_score(test_labels, predictions, labels=list(LABELS), average="weighted", zero_division=0)
        )
        independent_metrics["confusion_matrix"] = {
            "labels": list(LABELS),
            "matrix": confusion_matrix(test_labels, predictions, labels=list(LABELS)).tolist(),
        }
        model_results[configuration["id"]] = {
            "name": configuration["name"],
            "configuration": {
                "tfidf_ngram_range": list(configuration["ngram_range"]),
                "classifier": configuration["classifier"],
                "class_weight": None,
                "random_state": RANDOM_STATE,
            },
            "cross_validation": cv_results[configuration["id"]],
            "independent_test": independent_metrics,
            "experimental_artifacts": {
                "vectorizer": str(vectorizer_path.relative_to(BASE_DIRECTORY.parent)),
                "classifier": str(classifier_path.relative_to(BASE_DIRECTORY.parent)),
            },
        }

    old_results = json.loads(OLD_RESULTS_PATH.read_text(encoding="utf-8"))
    old_model_a = old_results["models"]["A"]["independent_test"]
    new_best_accuracy = max(result["independent_test"]["accuracy"] for result in model_results.values())
    new_best_macro_f1 = max(result["independent_test"]["macro_f1"] for result in model_results.values())
    protected_hashes_after = {
        name: sha256(path) for name, path in PROTECTED_PRODUCTION_PATHS.items()
    }
    test_hash_after = sha256(TEST_DATASET_PATH)
    if protected_hashes_before != protected_hashes_after or test_hash_before != test_hash_after:
        raise RuntimeError("A protected dataset or production artifact changed during the experiment")

    results = {
        "training_dataset": {
            "path": "backend/dataset/reviews_diverse_augmented.csv",
            "size": int(len(training_dataset)),
            "class_distribution": {label: int(distribution[label]) for label in LABELS},
            "sources": ["reviews_diverse.csv", "reviews_targeted.csv"],
        },
        "cross_validation_protocol": {
            "method": "StratifiedKFold",
            "folds": FOLD_COUNT,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "standard_deviation_ddof": 0,
            "locked_test_used_for_cv": False,
        },
        "independent_test": {
            "path": "backend/dataset/reviews_test.csv",
            "size": int(len(test_dataset)),
            "class_distribution": {label: int(test_distribution[label]) for label in LABELS},
            "used_for_training_or_tuning": False,
            "same_locked_set_for_both_models": True,
            "sha256_before_evaluation": test_hash_before,
            "sha256_after_evaluation": test_hash_after,
        },
        "old_diverse_model_a_baseline": {
            "accuracy": old_model_a["accuracy"],
            "macro_f1": old_model_a["macro_f1"],
        },
        "augmented_best_results": {
            "best_independent_accuracy": new_best_accuracy,
            "best_independent_macro_f1": new_best_macro_f1,
            "accuracy_improvement": new_best_accuracy - old_model_a["accuracy"],
            "macro_f1_improvement": new_best_macro_f1 - old_model_a["macro_f1"],
            "accuracy_improvement_percentage_points": 100 * (new_best_accuracy - old_model_a["accuracy"]),
            "macro_f1_improvement_percentage_points": 100 * (new_best_macro_f1 - old_model_a["macro_f1"]),
        },
        "protected_hashes_before": {**protected_hashes_before, "locked_test_dataset": test_hash_before},
        "protected_hashes_after": {**protected_hashes_after, "locked_test_dataset": test_hash_after},
        "protected_files_unchanged": True,
        "models": model_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_cv_table(cv_results)
    print_test_table(model_results)
    print("\nOld Model A baseline:", old_model_a)
    print("Best augmented metrics:", results["augmented_best_results"])
    print("Protected files unchanged:", results["protected_files_unchanged"])
    print("Saved results:", RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()