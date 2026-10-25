
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amazon_reviews_cleaned.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amazon_reviews_for_google_sheets.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Keep useful columns for review analysis.
    columns = [
        "review_id",
        "product_category",
        "review_title",
        "review_text",
        "cleaned_text",
        "sentiment",
        "source_label",
    ]

    columns = [column for column in columns if column in df.columns]
    export_df = df[columns].copy()

    export_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Google Sheets export created.")
    print(f"Rows: {len(export_df):,}")
    print(f"Columns: {columns}")
    print(f"File: {OUTPUT_FILE}")
    print("Import this CSV into Google Sheets using File > Import > Upload.")


if __name__ == "__main__":
    main()
