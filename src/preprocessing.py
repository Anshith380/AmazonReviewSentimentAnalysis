
import re
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "amazon_reviews_cleaned.csv"


def clean_text(text):
    """Normalize review text while preserving useful sentiment words."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)       # Remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # Remove URLs
    text = re.sub(r"[\r\n\t]+", " ", text)    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}\n"
            "Run the download script first."
        )

    logging.info("Loading raw reviews...")
    df = pd.read_csv(RAW_FILE)

    required_columns = {"review_text", "sentiment"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    initial_count = len(df)

    # Remove missing and empty review text.
    df = df.dropna(subset=["review_text", "sentiment"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]

    # Remove duplicate review text.
    df = df.drop_duplicates(subset=["review_text"]).copy()

    # Normalize text.
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    # Remove text that becomes empty after cleaning.
    df = df[df["cleaned_text"] != ""].copy()

    # Validate the existing source labels; do not invent new labels.
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()
    df = df[df["sentiment"].isin(["Positive", "Negative"])].copy()

    # Check for rows that cannot be used for training.
    df["word_count"] = df["cleaned_text"].str.split().str.len()
    df = df[df["word_count"] > 0].copy()

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nDATA CLEANING COMPLETED")
    print("-" * 40)
    print(f"Rows before cleaning: {initial_count:,}")
    print(f"Rows after cleaning:  {len(df):,}")
    print(f"Rows removed:         {initial_count - len(df):,}")
    print(f"Output file:           {OUTPUT_FILE}")

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    print("\nReview length statistics:")
    print(df["word_count"].describe().round(2).to_string())

    print("\nSample cleaned reviews:")
    print(df[["sentiment", "cleaned_text"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
