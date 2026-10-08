from pathlib import Path

import joblib

from services.sentiment import normalize_review


BASE_DIRECTORY = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIRECTORY / "models" / "experimental_diverse_vectorizer_a.joblib"
CLASSIFIER_PATH = BASE_DIRECTORY / "models" / "experimental_diverse_a.joblib"
REVIEWS = (
    "I am very happy with the plumber service",
    "The electrician arrived on time and fixed the problem",
    "The service was okay, nothing special",
    "The worker was late and the service was poor",
    "I am disappointed with the service",
)


def main() -> None:
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    print(f"Model A vectorizer: {VECTORIZER_PATH}")
    print(f"Model A classifier: {CLASSIFIER_PATH}")
    print("\nReview | Predicted sentiment | Confidence")
    for review in REVIEWS:
        features = vectorizer.transform([normalize_review(review)])
        probabilities = classifier.predict_proba(features)[0]
        best_index = probabilities.argmax()
        sentiment = str(classifier.classes_[best_index]).lower()
        confidence = float(probabilities[best_index])
        print(f"{review} | {sentiment} | {confidence:.4f}")


if __name__ == "__main__":
    main()