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
DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_diverse.csv"
TEST_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_test.csv"
PRODUCTION_VECTOR_PATH = BASE_DIRECTORY / "models" / "tfidf_vectorizer.joblib"
PRODUCTION_CLASSIFIER_PATH = BASE_DIRECTORY / "models" / "sentiment_model.joblib"
OLD_COMPARISON_PATH = BASE_DIRECTORY / "models" / "model_comparison.json"
RESULTS_PATH = BASE_DIRECTORY / "models" / "diverse_model_comparison.json"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
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


def artifact_paths(model_id: str) -> tuple[Path, Path]:
    model_id = model_id.lower()
    return (
        MODEL_DIRECTORY / f"experimental_diverse_vectorizer_{model_id}.joblib",
        MODEL_DIRECTORY / f"experimental_diverse_{model_id}.joblib",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_classifier(configuration: dict):
    if configuration["classifier"] == "linear_svc":
        return LinearSVC(random_state=RANDOM_STATE)
    return LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=configuration["class_weight"],
    )


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


def cross_validate(reviews: list[str], labels: list[str]) -> dict:
    splitter = StratifiedKFold(
        n_splits=FOLD_COUNT,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
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
            fold_metrics = calculate_metrics(fold_validation_labels, predictions)
            folds.append({"fold": fold_number, **fold_metrics})

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


def print_metrics_table(title: str, results: dict, result_key: str) -> None:
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
    print(f"{'Model':<5} {'Configuration':<55}" + "".join(f"{heading:>13}" for _, heading in columns))
    for configuration in CONFIGURATIONS:
        metrics = results[configuration["id"]][result_key]
        row = f"{configuration['id']:<5} {configuration['name']:<55}"
        row += "".join(f"{metrics[key]:>13.4f}" for key, _ in columns)
        print(row)


def print_cv_table(cv_results: dict) -> None:
    columns = (
        ("mean_accuracy", "Mean Acc"),
        ("std_accuracy", "Std Acc"),
        ("mean_macro_precision", "Macro P"),
        ("mean_macro_recall", "Macro R"),
        ("mean_macro_f1", "Mean Macro F1"),
        ("std_macro_f1", "Std Macro F1"),
        ("mean_weighted_f1", "Weighted F1"),
    )
    print("\n5-fold StratifiedKFold on diverse training data")
    print(f"{'Model':<5} {'Configuration':<55}" + "".join(f"{heading:>15}" for _, heading in columns))
    for configuration in CONFIGURATIONS:
        metrics = cv_results[configuration["id"]]
        row = f"{configuration['id']:<5} {configuration['name']:<55}"
        row += "".join(f"{metrics[key]:>15.4f}" for key, _ in columns)
        print(row)


def run_experiment() -> dict:
    output_paths = [RESULTS_PATH]
    for configuration in CONFIGURATIONS:
        output_paths.extend(artifact_paths(configuration["id"]))
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite experimental outputs: {existing_outputs}")

    protected_hashes_before = {
        "diverse_training_dataset": file_sha256(DATASET_PATH),
        "original_training_dataset": file_sha256(BASE_DIRECTORY / "dataset" / "reviews.csv"),
        "production_vectorizer": file_sha256(PRODUCTION_VECTOR_PATH),
        "production_classifier": file_sha256(PRODUCTION_CLASSIFIER_PATH),
    }

    training_dataset = read_dataset(DATASET_PATH)
    training_reviews = training_dataset["review"].map(normalize_review).tolist()
    training_labels = training_dataset["sentiment"].tolist()
    training_distribution = Counter(training_labels)
    if len(training_dataset) != 1500 or any(training_distribution[label] != 500 for label in LABELS):
        raise ValueError(f"Diverse dataset must contain 1,500 reviews, 500 per class: {dict(training_distribution)}")

    cv_results = cross_validate(training_reviews, training_labels)

    # The independent set is intentionally accessed only after cross-validation is complete.
    independent_test_hash_before = file_sha256(TEST_DATASET_PATH)
    independent_dataset = read_dataset(TEST_DATASET_PATH)
    independent_distribution = Counter(independent_dataset["sentiment"])
    expected_distribution = {label: 50 for label in LABELS}
    if len(independent_dataset) != 150 or dict(independent_distribution) != expected_distribution:
        raise ValueError(
            "Locked test set must contain exactly 150 reviews, 50 per class; "
            f"found {len(independent_dataset)} and {dict(independent_distribution)}"
        )
    independent_reviews = independent_dataset["review"].map(normalize_review).tolist()
    independent_labels = independent_dataset["sentiment"].tolist()

    final_results = {}
    for configuration in CONFIGURATIONS:
        vectorizer = TfidfVectorizer(
            ngram_range=configuration["ngram_range"],
            min_df=1,
        )
        training_features = vectorizer.fit_transform(training_reviews)
        classifier = create_classifier(configuration)
        classifier.fit(training_features, training_labels)

        vectorizer_path, classifier_path = artifact_paths(configuration["id"])
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)

        independent_features = vectorizer.transform(independent_reviews)
        predictions = classifier.predict(independent_features).tolist()
        metrics = calculate_metrics(independent_labels, predictions)
        metrics["confusion_matrix"] = {
            "labels": list(LABELS),
            "matrix": confusion_matrix(
                independent_labels,
                predictions,
                labels=list(LABELS),
            ).tolist(),
        }
        final_results[configuration["id"]] = {
            "name": configuration["name"],
            "configuration": {
                "tfidf_ngram_range": list(configuration["ngram_range"]),
                "classifier": configuration["classifier"],
                "class_weight": configuration["class_weight"],
                "random_state": RANDOM_STATE,
            },
            "cross_validation": cv_results[configuration["id"]],
            "independent_test": metrics,
            "experimental_artifacts": {
                "vectorizer": str(vectorizer_path.relative_to(BASE_DIRECTORY.parent)),
                "classifier": str(classifier_path.relative_to(BASE_DIRECTORY.parent)),
            },
        }

    old_results = json.loads(OLD_COMPARISON_PATH.read_text(encoding="utf-8"))
    old_best_accuracy = max(
        model["independent_test"]["accuracy"] for model in old_results["models"].values()
    )
    old_best_macro_f1 = max(
        model["independent_test"]["macro_f1"] for model in old_results["models"].values()
    )
    new_best_accuracy = max(model["independent_test"]["accuracy"] for model in final_results.values())
    new_best_macro_f1 = max(model["independent_test"]["macro_f1"] for model in final_results.values())

    protected_hashes_after = {
        "diverse_training_dataset": file_sha256(DATASET_PATH),
        "original_training_dataset": file_sha256(BASE_DIRECTORY / "dataset" / "reviews.csv"),
        "production_vectorizer": file_sha256(PRODUCTION_VECTOR_PATH),
        "production_classifier": file_sha256(PRODUCTION_CLASSIFIER_PATH),
    }
    if protected_hashes_before != protected_hashes_after:
        raise RuntimeError("A protected dataset or production model changed during the experiment")
    independent_test_hash_after = file_sha256(TEST_DATASET_PATH)
    if independent_test_hash_before != independent_test_hash_after:
        raise RuntimeError("The independent test dataset changed during the experiment")

    results = {
        "training_dataset": {
            "path": "backend/dataset/reviews_diverse.csv",
            "size": int(len(training_dataset)),
            "class_distribution": {label: int(training_distribution[label]) for label in LABELS},
        },
        "cross_validation_protocol": {
            "strategy": "StratifiedKFold",
            "folds": FOLD_COUNT,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "standard_deviation_ddof": 0,
            "dataset_used": "reviews_diverse.csv only",
        },
        "independent_test": {
            "path": "backend/dataset/reviews_test.csv",
            "size": int(len(independent_dataset)),
            "class_distribution": {label: int(independent_distribution[label]) for label in LABELS},
            "used_for_training_or_tuning": False,
            "same_test_set_for_all_models": True,
        },
        "old_training_baseline": {
            "source": "backend/models/model_comparison.json",
            "best_independent_accuracy": old_best_accuracy,
            "best_independent_macro_f1": old_best_macro_f1,
        },
        "new_diverse_training_results": {
            "best_independent_accuracy": new_best_accuracy,
            "best_independent_macro_f1": new_best_macro_f1,
            "accuracy_improvement": new_best_accuracy - old_best_accuracy,
            "macro_f1_improvement": new_best_macro_f1 - old_best_macro_f1,
        },
        "protected_file_hashes_before": {
            **protected_hashes_before,
            "independent_test_dataset": independent_test_hash_before,
        },
        "protected_file_hashes_after": {
            **protected_hashes_after,
            "independent_test_dataset": independent_test_hash_after,
        },
        "protected_files_unchanged": True,
        "models": final_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print_cv_table(cv_results)
    print_metrics_table("Independent test set (models trained on all diverse reviews)", final_results, "independent_test")
    print("\nIndependent-test confusion matrices (rows/columns: negative, neutral, positive)")
    for configuration in CONFIGURATIONS:
        matrix = final_results[configuration["id"]]["independent_test"]["confusion_matrix"]["matrix"]
        print(f"Model {configuration['id']}: {matrix}")
    print("\nOld training data best accuracy / macro F1:", old_best_accuracy, old_best_macro_f1)
    print("New diverse data best accuracy / macro F1:", new_best_accuracy, new_best_macro_f1)
    print("Independent accuracy improvement:", new_best_accuracy - old_best_accuracy)
    print("Independent macro F1 improvement:", new_best_macro_f1 - old_best_macro_f1)
    print("Protected datasets and production Joblib files unchanged:", protected_hashes_before == protected_hashes_after)
    print("Saved comparison:", RESULTS_PATH)
    return results


if __name__ == "__main__":
    run_experiment()