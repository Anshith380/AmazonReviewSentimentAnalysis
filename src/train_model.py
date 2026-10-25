
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "amazon_reviews_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"
METRICS_FILE = REPORT_DIR / "model_metrics.json"
REPORT_FILE = REPORT_DIR / "classification_report.txt"
CONFUSION_FILE = REPORT_DIR / "confusion_matrix.csv"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["cleaned_text", "sentiment"])

    X = df["cleaned_text"].astype(str)
    y = df["sentiment"].astype(str)

    if y.nunique() < 2:
        raise ValueError("Training requires at least two sentiment classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("Training sentiment model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "labels": labels,
        "classification_report": report,
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    REPORT_FILE.write_text(report, encoding="utf-8")

    pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ).to_csv(CONFUSION_FILE)

    print("\nMODEL TRAINING COMPLETED")
    print("-" * 40)
    print(f"Total reviews: {len(df):,}")
    print(f"Training reviews: {len(X_train):,}")
    print(f"Testing reviews: {len(X_test):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Model saved: {MODEL_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")
    print(f"Confusion matrix saved: {CONFUSION_FILE}")


if __name__ == "__main__":
    main()
