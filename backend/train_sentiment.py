from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from services.sentiment import normalize_review

BASE_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIRECTORY / "dataset" / "reviews.csv"
MODEL_DIRECTORY = BASE_DIRECTORY / "models"
VECTORIZER_PATH = MODEL_DIRECTORY / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIRECTORY / "sentiment_model.joblib"


def train() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError("Dataset must contain review and sentiment columns")

    dataset = dataset.dropna(subset=["review", "sentiment"]).copy()
    dataset["review"] = dataset["review"].map(normalize_review)
    dataset["sentiment"] = dataset["sentiment"].str.strip().str.lower()

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
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)

    print(f"Training samples: {len(training_reviews)}")
    print(f"Testing samples: {len(testing_reviews)}")
    print(f"Test accuracy: {accuracy_score(testing_labels, predictions):.4f}")
    print("Classification report:")
    print(classification_report(testing_labels, predictions, zero_division=0))
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved classifier: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    train()
