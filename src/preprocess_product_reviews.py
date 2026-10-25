
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "amazon_electronics_reviews.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "amazon_electronics_reviews_cleaned.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "product_id",
        "rating",
        "review_text",
        "sentiment_from_rating",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    before = len(df)

    df = df.dropna(
        subset=["product_id", "rating", "review_text"]
    ).copy()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df[
        df["product_id"].ne("")
        & df["review_text"].ne("")
        & df["rating"].between(1, 5)
    ].copy()

    df["cleaned_text"] = (
        df["review_text"]
        .str.lower()
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"http\S+|www\.\S+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[df["cleaned_text"].ne("")].copy()

    df = df.drop_duplicates(
        subset=["product_id", "review_text"]
    ).copy()

    df["sentiment_from_rating"] = df["rating"].apply(
        lambda rating: (
            "Positive" if rating >= 4
            else "Negative" if rating <= 2
            else "Neutral"
        )
    )

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nPRODUCT DATA PREPROCESSING COMPLETED")
    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())

    print("\nSentiment distribution:")
    print(df["sentiment_from_rating"].value_counts().to_string())


if __name__ == "__main__":
    main()
