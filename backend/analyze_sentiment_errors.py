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
VECTOR_PATH = BASE_DIRECTORY / "models" / "experimental_diverse_vectorizer_a.joblib"
CLASSIFIER_PATH = BASE_DIRECTORY / "models" / "experimental_diverse_a.joblib"
REPORT_PATH = BASE_DIRECTORY / "models" / "sentiment_error_analysis.json"
MISCLASSIFICATIONS_PATH = BASE_DIRECTORY / "models" / "sentiment_misclassifications.csv"
LABELS = ("positive", "neutral", "negative")

PROTECTED_PATHS = {
    "locked_test_dataset": TEST_DATASET_PATH,
    "original_training_dataset": BASE_DIRECTORY / "dataset" / "reviews.csv",
    "diverse_training_dataset": BASE_DIRECTORY / "dataset" / "reviews_diverse.csv",
    "production_classifier": BASE_DIRECTORY / "models" / "sentiment_model.joblib",
    "production_vectorizer": BASE_DIRECTORY / "models" / "tfidf_vectorizer.joblib",
    "experimental_classifier": CLASSIFIER_PATH,
    "experimental_vectorizer": VECTOR_PATH,
}

PATTERNS = {
    "negation": re.compile(
        r"\b(?:not|no|never|neither|nor|didn['’]?t|doesn['’]?t|wasn['’]?t|weren['’]?t|isn['’]?t|couldn['’]?t|wouldn['’]?t|cannot|can['’]?t)\b",
        re.IGNORECASE,
    ),
    "strong_positive_wording": re.compile(
        r"\b(?:excellent|amazing|wonderful|fantastic|outstanding|perfect(?:ly)?|brilliant|superb|very happy|really pleased|highly recommend)\b",
        re.IGNORECASE,
    ),
    "strong_negative_wording": re.compile(
        r"\b(?:terrible|disappointed|worst|awful|horrible|useless|appalling|dreadful|very poor|never again)\b",
        re.IGNORECASE,
    ),
    "informal_language": re.compile(
        r"\b(?:gonna|wanna|kinda|sorta|yeah|yep|nope|tho|pls|lol|tbh|okayish)\b|"
        r"\b(?:didn['’]t|doesn['’]t|wasn['’]t|weren['’]t|isn['’]t|couldn['’]t|wouldn['’]t)\b|"
        r"(?:!{2,}|\?{2,}|\.\.\.)",
        re.IGNORECASE,
    ),
    "ambiguous_or_hedged_wording": re.compile(
        r"\b(?:okay|fine|average|ordinary|nothing special|not bad|could be better|kind of|sort of|maybe|perhaps|a bit|somewhat|mixed|but)\b",
        re.IGNORECASE,
    ),
}

POSITIVE_TERMS = re.compile(
    r"\b(?:good|great|excellent|amazing|wonderful|fantastic|outstanding|perfect|brilliant|superb|happy|pleased|satisfied|helpful|friendly|professional|reliable|quickly|promptly|clean|neat|fair|reasonable|recommend|worthwhile|impressed)\b",
    re.IGNORECASE,
)
NEGATIVE_TERMS = re.compile(
    r"\b(?:bad|poor|terrible|awful|horrible|worst|disappointed|late|delay(?:ed)?|missed|failed|unfinished|incomplete|ignored|rude|careless|dirty|broken|overpriced|expensive|unhelpful|regret|problem|issue|complaint|cancelled|canceled)\b",
    re.IGNORECASE,
)
FACTUAL_TERMS = re.compile(
    r"\b(?:arrived|replaced|completed|checked|recorded|measured|installed|visited|inspected|scheduled|listed|invoice|receipt|estimate|appointment|at \d|am|pm|delivered|collected|changed|repaired|provided)\b",
    re.IGNORECASE,
)
SERVICE_TERMS = re.compile(
    r"\b(?:plumb(?:er|ing)?|tap|faucet|pipe|leak|drain|toilet|cistern|electric(?:al|ian)?|wiring|switch|breaker|outlet|clean(?:er|ing)?|carpent(?:er|ry)|wood|hinge|shelf|air[- ]?condition(?:er|ing)|\bac\b|refrigerator|fridge|washer|washing machine|dryer|dishwasher|oven|appliance|paint(?:er|ing)?|scooter|vehicle|brake|battery|mechanic|food|meal|vendor|grocery|shop|delivery|parcel|gutter|handrail|maintenance)\b",
    re.IGNORECASE,
)

ERROR_BUCKETS = (
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


def analyze_patterns(review: str, confidence: float) -> dict[str, bool]:
    positive_signal = bool(POSITIVE_TERMS.search(review))
    negative_signal = bool(NEGATIVE_TERMS.search(review))
    word_count = len(review.split())
    return {
        "negation": bool(PATTERNS["negation"].search(review)),
        "mixed_sentiment_terms": positive_signal and negative_signal,
        "strong_positive_wording": bool(PATTERNS["strong_positive_wording"].search(review)),
        "strong_negative_wording": bool(PATTERNS["strong_negative_wording"].search(review)),
        "neutral_factual_wording": (
            bool(FACTUAL_TERMS.search(review)) and not positive_signal and not negative_signal
        ),
        "short_review_1_to_10_words": 1 <= word_count <= 10,
        "long_review_36_or_more_words": word_count >= 36,
        "informal_language": bool(PATTERNS["informal_language"].search(review)),
        "service_specific_terminology": bool(SERVICE_TERMS.search(review)),
        "ambiguous_or_hedged_wording": bool(PATTERNS["ambiguous_or_hedged_wording"].search(review)),
        "confidence_below_0_60": confidence < 0.60,
    }


def main() -> dict:
    if REPORT_PATH.exists() or MISCLASSIFICATIONS_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing error-analysis report or CSV")

    protected_hashes_before = {
        name: sha256(path) for name, path in PROTECTED_PATHS.items()
    }

    dataset = pd.read_csv(TEST_DATASET_PATH)
    if list(dataset.columns) != ["review", "sentiment"]:
        raise ValueError("Test dataset must contain exactly review,sentiment columns")
    dataset["review"] = dataset["review"].fillna("").astype(str).str.strip()
    dataset["sentiment"] = dataset["sentiment"].fillna("").astype(str).str.strip().str.lower()
    if dataset["review"].eq("").any() or not dataset["sentiment"].isin(LABELS).all():
        raise ValueError("Test dataset contains blank reviews or invalid sentiment labels")

    expected_counts = {"positive": 50, "neutral": 50, "negative": 50}
    actual_counts = {
        label: int((dataset["sentiment"] == label).sum()) for label in LABELS
    }
    if len(dataset) != 150 or actual_counts != expected_counts:
        raise ValueError(
            f"Locked test dataset must have 150 reviews and 50 per class; got {len(dataset)}, {actual_counts}"
        )

    vectorizer = joblib.load(VECTOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    reviews = dataset["review"].tolist()
    actual = dataset["sentiment"].tolist()
    features = vectorizer.transform([normalize_review(review) for review in reviews])
    probability_rows = classifier.predict_proba(features)
    predictions = classifier.predict(features).tolist()
    class_indices = {str(label).lower(): index for index, label in enumerate(classifier.classes_)}

    misclassifications = []
    for review, actual_label, predicted_label, probabilities in zip(
        reviews, actual, predictions, probability_rows
    ):
        predicted_label = str(predicted_label).lower()
        confidence = float(probabilities[class_indices[predicted_label]])
        if actual_label != predicted_label:
            misclassifications.append(
                {
                    "review": review,
                    "actual_sentiment": actual_label,
                    "predicted_sentiment": predicted_label,
                    "confidence": round(confidence, 6),
                }
            )

    total = len(dataset)
    incorrect_count = len(misclassifications)
    correct_count = total - incorrect_count
    error_breakdown = {
        name: sum(
            error["actual_sentiment"] == actual_label
            and error["predicted_sentiment"] == predicted_label
            for error in misclassifications
        )
        for name, actual_label, predicted_label in ERROR_BUCKETS
    }
    length_breakdown = {}
    for name, minimum, maximum in LENGTH_BUCKETS:
        rows_in_bucket = [
            len(review.split())
            for review in reviews
            if len(review.split()) >= minimum
            and (maximum is None or len(review.split()) <= maximum)
        ]
        errors_in_bucket = sum(
            minimum <= len(error["review"].split())
            and (maximum is None or len(error["review"].split()) <= maximum)
            for error in misclassifications
        )
        length_breakdown[name] = {
            "total_reviews": len(rows_in_bucket),
            "error_count": int(errors_in_bucket),
            "error_percentage": round(100 * errors_in_bucket / len(rows_in_bucket), 4)
            if rows_in_bucket
            else 0.0,
        }

    pattern_counts = Counter()
    pattern_examples = {}
    for error in misclassifications:
        flags = analyze_patterns(error["review"], error["confidence"])
        error["pattern_flags"] = flags
        for pattern_name, matched in flags.items():
            if matched:
                pattern_counts[pattern_name] += 1
                pattern_examples.setdefault(pattern_name, [])
                if len(pattern_examples[pattern_name]) < 5:
                    pattern_examples[pattern_name].append(error["review"])

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
    report = {
        "model": {
            "vectorizer": str(VECTOR_PATH.relative_to(BASE_DIRECTORY.parent)),
            "classifier": str(CLASSIFIER_PATH.relative_to(BASE_DIRECTORY.parent)),
        },
        "test_dataset": {
            "path": "backend/dataset/reviews_test.csv",
            "total_reviews": total,
            "class_counts": actual_counts,
        },
        "metrics": {
            "correctly_classified_reviews": correct_count,
            "incorrectly_classified_reviews": incorrect_count,
            "overall_accuracy": round(correct_count / total, 6),
            "error_count": incorrect_count,
            "error_percentage": round(100 * incorrect_count / total, 4),
        },
        "error_breakdown": error_breakdown,
        "confusion_matrix": confusion,
        "error_counts_by_review_length": length_breakdown,
        "pattern_analysis": {
            "method": "Transparent case-insensitive keyword/phrase heuristics applied to misclassified review text; flags may overlap and are descriptive, not causal explanations.",
            "counts": {name: int(pattern_counts[name]) for name in PATTERNS.keys()} | {
                "mixed_sentiment_terms": int(pattern_counts["mixed_sentiment_terms"]),
                "neutral_factual_wording": int(pattern_counts["neutral_factual_wording"]),
                "short_review_1_to_10_words": int(pattern_counts["short_review_1_to_10_words"]),
                "long_review_36_or_more_words": int(pattern_counts["long_review_36_or_more_words"]),
                "service_specific_terminology": int(pattern_counts["service_specific_terminology"]),
                "confidence_below_0_60": int(pattern_counts["confidence_below_0_60"]),
            },
            "examples": pattern_examples,
        },
        "misclassifications": misclassifications,
    }

    with MISCLASSIFICATIONS_PATH.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["review", "actual_sentiment", "predicted_sentiment", "confidence"],
        )
        writer.writeheader()
        writer.writerows(
            {key: error[key] for key in writer.fieldnames}
            for error in misclassifications
        )
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    protected_hashes_after = {
        name: sha256(path) for name, path in PROTECTED_PATHS.items()
    }
    if protected_hashes_before != protected_hashes_after:
        raise RuntimeError("A protected dataset or model changed during error analysis")
    report["protected_file_hashes_before"] = protected_hashes_before
    report["protected_file_hashes_after"] = protected_hashes_after
    report["protected_files_unchanged"] = True
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Total test reviews: {total}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")
    print(f"Accuracy: {correct_count / total:.6f}")
    print(f"Error percentage: {100 * incorrect_count / total:.4f}%")
    print("Error breakdown:")
    for name, count in error_breakdown.items():
        print(f"  {name}: {count}")
    print("Error counts by review length:")
    for name, values in length_breakdown.items():
        print(f"  {name}: {values['error_count']} / {values['total_reviews']}")
    print("Pattern counts:")
    for name, count in report["pattern_analysis"]["counts"].items():
        print(f"  {name}: {count}")
    print("Misclassified reviews:")
    for error in misclassifications:
        print(
            f"  actual={error['actual_sentiment']} predicted={error['predicted_sentiment']} "
            f"confidence={error['confidence']:.4f} | {error['review']}"
        )
    print(f"Saved analysis: {REPORT_PATH}")
    print(f"Saved misclassifications: {MISCLASSIFICATIONS_PATH}")
    print("Protected files unchanged:", protected_hashes_before == protected_hashes_after)
    return report


if __name__ == "__main__":
    main()