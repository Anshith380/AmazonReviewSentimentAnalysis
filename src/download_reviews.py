
from pathlib import Path
import logging

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REVIEWS = 10000


def main():
    print("Connecting to the public Amazon review dataset...")

    dataset = load_dataset(
        "fancyzhx/amazon_polarity",
        split="train",
        streaming=True,
    )

    rows = []

    for i, review in enumerate(dataset):
        title = str(review.get("title") or "").strip()
        content = str(review.get("content") or "").strip()
        text = f"{title}. {content}".strip(". ")

        if not text:
            continue

        label = int(review["label"])

        rows.append({
            "review_id": len(rows) + 1,
            "product_category": "Mixed Amazon Products",
            "review_title": title,
            "review_text": text,
            "sentiment": "Positive" if label == 1 else "Negative",
            "source_label": label,
        })

        if len(rows) >= TARGET_REVIEWS:
            break

        if (i + 1) % 1000 == 0:
            logging.info("Processed %s source records", i + 1)

    if not rows:
        raise RuntimeError("No review records were collected.")

    df = pd.DataFrame(rows)

    output_path = RAW_DIR / "amazon_reviews_raw.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDownload completed successfully!")
    print(f"Reviews saved: {len(df):,}")
    print(f"CSV file: {output_path}")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print("\nSample reviews:")
    print(df[["review_title", "sentiment", "review_text"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
