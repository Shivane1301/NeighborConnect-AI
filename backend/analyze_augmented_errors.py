import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
TEST_DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews_test.csv"
VECTOR_PATH = BASE_DIRECTORY / "models" / "experimental_augmented_vectorizer_d.joblib"
CLASSIFIER_PATH = BASE_DIRECTORY / "models" / "experimental_augmented_d.joblib"
OLD_ANALYSIS_PATH = BASE_DIRECTORY / "models" / "sentiment_error_analysis.json"
COMPARISON_PATH = BASE_DIRECTORY / "models" / "augmented_model_comparison.json"
MISCLASSIFICATIONS_PATH = BASE_DIRECTORY / "models" / "augmented_misclassifications.csv"
REPORT_PATH = BASE_DIRECTORY / "models" / "augmented_error_analysis.json"
LABELS = ("negative", "neutral", "positive")

PROTECTED_PATHS = {
    "locked_test_dataset": TEST_DATASET_PATH,
    "original_training_dataset": BASE_DIRECTORY / "dataset" / "reviews.csv",
    "diverse_training_dataset": BASE_DIRECTORY / "dataset" / "reviews_diverse.csv",
    "targeted_training_dataset": BASE_DIRECTORY / "dataset" / "reviews_targeted.csv",
    "augmented_training_dataset": BASE_DIRECTORY / "dataset" / "reviews_diverse_augmented.csv",
    "production_classifier": BASE_DIRECTORY / "models" / "sentiment_model.joblib",
    "production_vectorizer": BASE_DIRECTORY / "models" / "tfidf_vectorizer.joblib",
    "experimental_classifier_d": CLASSIFIER_PATH,
    "experimental_vectorizer_d": VECTOR_PATH,
}

STRONG_POSITIVE = re.compile(
    r"\b(?:excellent|amazing|wonderful|fantastic|outstanding|perfect(?:ly)?|brilliant|superb|very happy|really pleased|highly recommend)\b",
    re.IGNORECASE,
)
STRONG_NEGATIVE = re.compile(
    r"\b(?:terrible|disappointed|worst|awful|horrible|useless|appalling|dreadful|very poor|never again)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|didn['’]?t|doesn['’]?t|wasn['’]?t|weren['’]?t|isn['’]?t|couldn['’]?t|wouldn['’]?t|cannot|can['’]?t)\b",
    re.IGNORECASE,
)
POSITIVE_TERMS = re.compile(
    r"\b(?:good|great|excellent|amazing|wonderful|fantastic|outstanding|perfect|brilliant|superb|happy|pleased|satisfied|helpful|friendly|professional|reliable|quickly|promptly|clean|neat|fair|reasonable|recommend|worthwhile|impressed)\b",
    re.IGNORECASE,
)
NEGATIVE_TERMS = re.compile(
    r"\b(?:bad|poor|terrible|awful|horrible|worst|disappointed|late|delay(?:ed)?|missed|failed|unfinished|incomplete|ignored|rude|careless|dirty|broken|overpriced|expensive|unhelpful|regret|problem|issue|complaint|cancelled|canceled|damaged)\b",
    re.IGNORECASE,
)
FACTUAL_TERMS = re.compile(
    r"\b(?:arrived|replaced|completed|checked|recorded|measured|installed|visited|inspected|scheduled|listed|invoice|receipt|estimate|appointment|delivered|collected|changed|repaired|provided|fitted|tested|at \d|am|pm)\b",
    re.IGNORECASE,
)
INFORMAL = re.compile(
    r"\b(?:gonna|wanna|kinda|sorta|yeah|yep|nope|tho|pls|lol|tbh|okayish)\b|"
    r"\b(?:didn['’]t|doesn['’]t|wasn['’]t|weren['’]t|isn['’]t|couldn['’]t|wouldn['’]t)\b|"
    r"(?:!{2,}|\?{2,}|\.\.\.)",
    re.IGNORECASE,
)
AMBIGUOUS = re.compile(
    r"\b(?:okay|fine|average|ordinary|nothing special|not bad|could be better|kind of|sort of|maybe|perhaps|a bit|somewhat|mixed|but|although|though)\b",
    re.IGNORECASE,
)
SERVICE_TERMS = re.compile(
    r"\b(?:plumb(?:er|ing)?|tap|faucet|pipe|leak|drain|toilet|cistern|electric(?:al|ian)?|wiring|switch|breaker|outlet|clean(?:er|ing)?|carpent(?:er|ry)|wood|hinge|shelf|air[- ]?condition(?:er|ing)|ac|refrigerator|fridge|washer|washing machine|dryer|dishwasher|oven|appliance|paint(?:er|ing)?|scooter|vehicle|brake|battery|mechanic|food|meal|vendor|grocery|shop|delivery|parcel|gutter|handrail|maintenance)\b",
    re.IGNORECASE,
)

ERROR_CATEGORIES = (
    ("positive_to_negative", "positive", "negative"),
    ("positive_to_neutral", "positive", "neutral"),
    ("neutral_to_positive", "neutral", "positive"),
    ("neutral_to_negative", "neutral", "negative"),
    ("negative_to_positive", "negative", "positive"),
    ("negative_to_neutral", "negative", "neutral"),
)

LENGTH_BUCKETS = (
    ("1-10_words", 1, 10),
    ("11-20_words", 11, 20),
    ("21-35_words", 21, 35),
    ("36+_words", 36, None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error_flags(review: str, actual: str, predicted: str, confidence: float | None) -> dict:
    positive_signal = bool(POSITIVE_TERMS.search(review))
    negative_signal = bool(NEGATIVE_TERMS.search(review))
    word_count = len(review.split())
    return {
        "negation": bool(NEGATION.search(review)),
        "mixed_sentiment_terms": positive_signal and negative_signal,
        "strong_positive_wording": bool(STRONG_POSITIVE.search(review)),
        "strong_negative_wording": bool(STRONG_NEGATIVE.search(review)),
        "neutral_factual_wording": (
            actual == "neutral"
            and bool(FACTUAL_TERMS.search(review))
            and not positive_signal
            and not negative_signal
        ),
        "short_review_1_to_10_words": 1 <= word_count <= 10,
        "long_review_36_or_more_words": word_count >= 36,
        "service_specific_terminology": bool(SERVICE_TERMS.search(review)),
        "informal_wording": bool(INFORMAL.search(review)),
        "ambiguous_or_hedged_wording": bool(AMBIGUOUS.search(review)),
        "strong_positive_predicted_negative": (
            bool(STRONG_POSITIVE.search(review)) and predicted == "negative"
        ),
        "strong_negative_predicted_positive": (
            bool(STRONG_NEGATIVE.search(review)) and predicted == "positive"
        ),
        "neutral_factual_predicted_polar": (
            actual == "neutral"
            and predicted in ("positive", "negative")
            and bool(FACTUAL_TERMS.search(review))
            and not positive_signal
            and not negative_signal
        ),
        "confidence_available": confidence is not None,
    }


def comparison_status(old_count: int, new_count: int) -> str:
    if new_count < old_count:
        return "improved"
    if new_count > old_count:
        return "became_worse"
    return "stayed_the_same"


def main() -> dict:
    if REPORT_PATH.exists() or MISCLASSIFICATIONS_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing augmented error-analysis output")
    protected_hashes_before = {name: sha256(path) for name, path in PROTECTED_PATHS.items()}

    test_dataset = pd.read_csv(TEST_DATASET_PATH)
    if list(test_dataset.columns) != ["review", "sentiment"]:
        raise ValueError("Locked test dataset must have exactly review,sentiment columns")
    test_dataset["review"] = test_dataset["review"].fillna("").astype(str).str.strip()
    test_dataset["sentiment"] = test_dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    if test_dataset["review"].eq("").any() or not test_dataset["sentiment"].isin(LABELS).all():
        raise ValueError("Locked test dataset contains blank reviews or invalid sentiment labels")
    class_counts = {label: int((test_dataset["sentiment"] == label).sum()) for label in LABELS}
    if len(test_dataset) != 150 or class_counts != {"negative": 50, "neutral": 50, "positive": 50}:
        raise ValueError(f"Expected 150 locked test reviews, 50 per class; got {len(test_dataset)}, {class_counts}")

    vectorizer = joblib.load(VECTOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    reviews = test_dataset["review"].tolist()
    actual = test_dataset["sentiment"].tolist()
    features = vectorizer.transform([normalize_review(review) for review in reviews])
    predictions = [str(label).lower() for label in classifier.predict(features)]

    decision_values = classifier.decision_function(features)
    if getattr(decision_values, "ndim", 1) == 1:
        decision_values = decision_values.reshape(-1, 1)
    errors = []
    for index, (review, actual_label, predicted_label) in enumerate(zip(reviews, actual, predictions)):
        if actual_label == predicted_label:
            continue
        row_scores = decision_values[index]
        ordered_scores = sorted((float(score) for score in row_scores), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else abs(ordered_scores[0])
        errors.append(
            {
                "review": review,
                "actual_sentiment": actual_label,
                "predicted_sentiment": predicted_label,
                "confidence": None,
                "confidence_note": "LinearSVC does not provide calibrated predict_proba confidence.",
                "decision_margin": round(margin, 6),
            }
        )

    total = len(reviews)
    correct_count = total - len(errors)
    accuracy = correct_count / total
    old_report = json.loads(OLD_ANALYSIS_PATH.read_text(encoding="utf-8"))
    old_breakdown = old_report["error_breakdown"]
    new_breakdown = {
        name: sum(
            error["actual_sentiment"] == actual_label
            and error["predicted_sentiment"] == predicted_label
            for error in errors
        )
        for name, actual_label, predicted_label in ERROR_CATEGORIES
    }
    category_comparison = {
        name: {
            "old_model_a_count": int(old_breakdown[name]),
            "new_model_d_count": int(new_breakdown[name]),
            "change": comparison_status(int(old_breakdown[name]), int(new_breakdown[name])),
            "difference_new_minus_old": int(new_breakdown[name]) - int(old_breakdown[name]),
        }
        for name, _, _ in ERROR_CATEGORIES
    }

    length_breakdown = {}
    for name, minimum, maximum in LENGTH_BUCKETS:
        bucket_reviews = [
            review
            for review in reviews
            if len(review.split()) >= minimum
            and (maximum is None or len(review.split()) <= maximum)
        ]
        bucket_errors = sum(
            len(error["review"].split()) >= minimum
            and (maximum is None or len(error["review"].split()) <= maximum)
            for error in errors
        )
        length_breakdown[name] = {
            "total_reviews": len(bucket_reviews),
            "error_count": int(bucket_errors),
            "error_percentage": round(100 * bucket_errors / len(bucket_reviews), 4)
            if bucket_reviews
            else 0.0,
        }

    pattern_names = (
        "negation",
        "mixed_sentiment_terms",
        "strong_positive_wording",
        "strong_negative_wording",
        "neutral_factual_wording",
        "short_review_1_to_10_words",
        "long_review_36_or_more_words",
        "service_specific_terminology",
        "informal_wording",
        "ambiguous_or_hedged_wording",
    )
    pattern_counts = Counter()
    pattern_examples = {name: [] for name in pattern_names}
    concerns = {
        "strong_positive_predicted_negative": [],
        "strong_negative_predicted_positive": [],
        "neutral_factual_predicted_polar": [],
    }
    for error in errors:
        flags = error_flags(
            error["review"],
            error["actual_sentiment"],
            error["predicted_sentiment"],
            error["confidence"],
        )
        error["pattern_flags"] = flags
        for name in pattern_names:
            if flags[name]:
                pattern_counts[name] += 1
                if len(pattern_examples[name]) < 5:
                    pattern_examples[name].append(error["review"])
        for concern_name in concerns:
            if flags[concern_name]:
                concerns[concern_name].append(
                    {
                        "review": error["review"],
                        "actual_sentiment": error["actual_sentiment"],
                        "predicted_sentiment": error["predicted_sentiment"],
                        "confidence": None,
                        "decision_margin": error["decision_margin"],
                    }
                )

    confusion = {
        "labels": list(LABELS),
        "matrix": [
            [
                sum(actual_label == row_label and predicted_label == column_label
                    for actual_label, predicted_label in zip(actual, predictions))
                for column_label in LABELS
            ]
            for row_label in LABELS
        ],
    }
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    model_metrics = comparison["models"]["D"]["independent_test"]
    report = {
        "model": {
            "vectorizer": str(VECTOR_PATH.relative_to(BASE_DIRECTORY.parent)),
            "classifier": str(CLASSIFIER_PATH.relative_to(BASE_DIRECTORY.parent)),
            "classifier_type": type(classifier).__name__,
            "confidence_available": False,
            "confidence_note": "LinearSVC has no calibrated predict_proba output; confidence fields are null and raw decision margins are reported separately.",
        },
        "test_dataset": {
            "path": "backend/dataset/reviews_test.csv",
            "total_reviews": total,
            "class_counts": class_counts,
        },
        "metrics": {
            "correct_predictions": correct_count,
            "incorrect_predictions": len(errors),
            "accuracy": accuracy,
            "error_count": len(errors),
            "error_percentage": 100 * len(errors) / total,
            "macro_f1": model_metrics["macro_f1"],
        },
        "confusion_matrix": confusion,
        "old_model_a_error_count": int(old_report["metrics"]["error_count"]),
        "new_model_d_error_count": len(errors),
        "error_count_difference_new_minus_old": len(errors) - int(old_report["metrics"]["error_count"]),
        "error_category_comparison": category_comparison,
        "error_counts_by_review_length": length_breakdown,
        "error_pattern_analysis": {
            "method": "Descriptive case-insensitive keyword and phrase checks on actual misclassified review text; flags may overlap and are not causal explanations.",
            "counts": {name: int(pattern_counts[name]) for name in pattern_names},
            "examples": pattern_examples,
        },
        "concerning_examples": concerns,
        "misclassifications": errors,
    }

    with MISCLASSIFICATIONS_PATH.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["review", "actual_sentiment", "predicted_sentiment", "confidence"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "review": error["review"],
                "actual_sentiment": error["actual_sentiment"],
                "predicted_sentiment": error["predicted_sentiment"],
                "confidence": "",
            }
            for error in errors
        )

    protected_hashes_after = {name: sha256(path) for name, path in PROTECTED_PATHS.items()}
    if protected_hashes_before != protected_hashes_after:
        raise RuntimeError("A protected dataset or model changed during analysis")
    report["protected_hashes_before"] = protected_hashes_before
    report["protected_hashes_after"] = protected_hashes_after
    report["protected_files_unchanged"] = True
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Test reviews: {total}; correct: {correct_count}; incorrect: {len(errors)}")
    print(f"Accuracy: {accuracy:.6f}; macro F1: {model_metrics['macro_f1']:.6f}; error rate: {100 * len(errors) / total:.4f}%")
    print("Confusion matrix [negative, neutral, positive]:", confusion["matrix"])
    print("Old vs new error category counts:")
    for name, values in category_comparison.items():
        print(f"  {name}: {values['old_model_a_count']} -> {values['new_model_d_count']} ({values['change']})")
    print("New error pattern counts:", dict(pattern_counts))
    print("Length errors:", length_breakdown)
    print("Concerning examples:", json.dumps(concerns, ensure_ascii=True))
    print("Misclassifications saved:", len(errors), MISCLASSIFICATIONS_PATH)
    print("Confidence: unavailable for LinearSVC; decision margins are in JSON only.")
    print("Protected files unchanged:", protected_hashes_before == protected_hashes_after)
    print("Saved analysis:", REPORT_PATH)
    return report


if __name__ == "__main__":
    main()