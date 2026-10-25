
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_electronics_reviews_cleaned.csv"
)


def load_data():
    assert DATA_FILE.exists(), f"Dataset not found: {DATA_FILE}"
    return pd.read_csv(DATA_FILE)


def test_product_dataset_is_not_empty():
    df = load_data()
    assert len(df) > 0


def test_required_columns_exist():
    df = load_data()
    required = {
        "product_id",
        "rating",
        "review_text",
        "sentiment_from_rating",
        "cleaned_text",
    }
    assert required.issubset(df.columns)


def test_ratings_are_between_one_and_five():
    df = load_data()
    assert df["rating"].between(1, 5).all()


def test_sentiment_matches_star_rating():
    df = load_data()

    expected = df["rating"].apply(
        lambda rating: (
            "Positive" if rating >= 4
            else "Negative" if rating <= 2
            else "Neutral"
        )
    )

    assert df["sentiment_from_rating"].equals(expected)


def test_reviews_have_product_ids():
    df = load_data()
    assert df["product_id"].notna().all()
    assert df["product_id"].astype(str).str.strip().ne("").all()


def test_review_text_is_not_empty():
    df = load_data()
    assert df["review_text"].notna().all()
    assert df["review_text"].astype(str).str.strip().ne("").all()


def test_cleaned_text_is_not_empty():
    df = load_data()
    assert df["cleaned_text"].notna().all()
    assert df["cleaned_text"].astype(str).str.strip().ne("").all()
