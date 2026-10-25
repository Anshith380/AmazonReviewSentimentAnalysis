
import gzip
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Electronics_5.json.gz"
)

DOWNLOAD_FILE = RAW_DIR / "electronics_reviews.json.gz"
PARTIAL_FILE = RAW_DIR / "electronics_reviews.json.gz.part"
OUTPUT_FILE = RAW_DIR / "amazon_electronics_reviews.csv"

TARGET_REVIEWS = 10000
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 10
TOTAL_SIZE = 495854086


# --------------------------------------------------
# 2. Download with resume and retry support
# --------------------------------------------------

def download_file():
    if DOWNLOAD_FILE.exists():
        logging.info("Compressed dataset already exists.")
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": "AmazonReviewSentimentAnalysis/1.0"
    })

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            downloaded = (
                PARTIAL_FILE.stat().st_size
                if PARTIAL_FILE.exists()
                else 0
            )

            if downloaded > TOTAL_SIZE:
                logging.warning(
                    "Partial file is unexpectedly large. Restarting."
                )
                PARTIAL_FILE.unlink()
                downloaded = 0

            headers = {}
            if downloaded:
                headers["Range"] = f"bytes={downloaded}-"

            logging.info(
                "Attempt %s/%s | Downloaded %.1f / %.1f MB",
                attempt,
                MAX_ATTEMPTS,
                downloaded / 1048576,
                TOTAL_SIZE / 1048576,
            )

            try:
                with session.get(
                    URL,
                    headers=headers,
                    stream=True,
                    timeout=(30, 120),
                ) as response:

                    if downloaded and response.status_code == 206:
                        content_range = response.headers.get(
                            "Content-Range", ""
                        )

                        expected_prefix = f"bytes {downloaded}-"
                        if not content_range.startswith(expected_prefix):
                            raise RuntimeError(
                                "Server returned an unexpected byte range: "
                                f"{content_range}"
                            )

                        mode = "ab"

                    elif response.status_code == 200:
                        # The server ignored Range. Restart safely.
                        downloaded = 0
                        mode = "wb"

                    else:
                        response.raise_for_status()
                        raise RuntimeError(
                            f"Unexpected HTTP status: "
                            f"{response.status_code}"
                        )

                    with open(PARTIAL_FILE, mode) as file:
                        for chunk in response.iter_content(
                            chunk_size=CHUNK_SIZE
                        ):
                            if not chunk:
                                continue

                            file.write(chunk)
                            downloaded += len(chunk)

                            percent = min(
                                downloaded / TOTAL_SIZE * 100,
                                100,
                            )

                            print(
                                f"\rDownload progress: {percent:.1f}% "
                                f"({downloaded / 1048576:.1f} MB)",
                                end="",
                                flush=True,
                            )

                print()

                actual_size = PARTIAL_FILE.stat().st_size

                if actual_size < TOTAL_SIZE:
                    logging.warning(
                        "Connection ended before completion. "
                        "The next attempt will resume."
                    )
                    continue

                if actual_size > TOTAL_SIZE:
                    raise RuntimeError(
                        f"Unexpected file size: {actual_size} bytes"
                    )

                # Validate the entire gzip stream, including its checksum.
                logging.info("Validating downloaded gzip file...")

                with gzip.open(PARTIAL_FILE, "rb") as compressed:
                    while compressed.read(8 * 1024 * 1024):
                        pass

                PARTIAL_FILE.replace(DOWNLOAD_FILE)

                logging.info("Download completed and validated.")
                logging.info("Saved to: %s", DOWNLOAD_FILE)
                return

            except (requests.RequestException, OSError, EOFError) as exc:
                logging.warning("Download interrupted: %s", exc)

            if attempt < MAX_ATTEMPTS:
                wait_seconds = min(attempt * 3, 20)
                logging.info(
                    "Retrying in %s seconds...",
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        raise RuntimeError(
            "Download did not complete after all attempts. "
            f"Partial file retained at: {PARTIAL_FILE}"
        )

    finally:
        session.close()


# --------------------------------------------------
# 3. Extract and validate review records
# --------------------------------------------------

def extract_reviews():
    if not DOWNLOAD_FILE.exists():
        raise FileNotFoundError(
            f"Compressed dataset not found: {DOWNLOAD_FILE}"
        )

    logging.info("Extracting Electronics reviews...")

    rows = []
    source_rows = 0

    with gzip.open(
        DOWNLOAD_FILE,
        mode="rt",
        encoding="utf-8",
    ) as file:

        for line in file:
            source_rows += 1

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            review_text = str(
                review.get("reviewText") or ""
            ).strip()

            asin = str(review.get("asin") or "").strip()
            rating = review.get("overall")

            if not review_text or not asin or rating is None:
                continue

            try:
                rating = float(rating)
            except (ValueError, TypeError):
                continue

            if not 1 <= rating <= 5:
                continue

            helpful = review.get("helpful", [0, 0])
            helpful_votes = 0

            if isinstance(helpful, list) and helpful:
                try:
                    helpful_votes = int(helpful[0])
                except (ValueError, TypeError):
                    helpful_votes = 0

            rows.append({
                "product_category": "Electronics",
                "product_id": asin,
                "rating": rating,
                "review_title": str(
                    review.get("summary") or ""
                ).strip(),
                "review_text": review_text,
                "review_date": str(
                    review.get("reviewTime") or ""
                ),
                "reviewer_id": str(
                    review.get("reviewerID") or ""
                ),
                "helpful_votes": helpful_votes,
            })

            if len(rows) >= TARGET_REVIEWS:
                break

            if source_rows % 100000 == 0:
                logging.info(
                    "Source rows read: %s | Valid reviews: %s",
                    source_rows,
                    len(rows),
                )

    if not rows:
        raise RuntimeError(
            "No valid review records were extracted."
        )

    df = pd.DataFrame(rows)

    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=["product_id", "review_text"]
    ).copy()

    # Derive sentiment labels from original star ratings.
    df["sentiment_from_rating"] = df["rating"].apply(
        lambda rating: (
            "Positive" if rating >= 4
            else "Negative" if rating <= 2
            else "Neutral"
        )
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 45)
    print("PRODUCT REVIEW EXTRACTION COMPLETED")
    print("=" * 45)
    print(f"Valid reviews collected: {before_deduplication:,}")
    print(f"Unique reviews saved: {len(df):,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Duplicate reviews removed: "
          f"{before_deduplication - len(df):,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nStar rating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())

    print("\nSentiment distribution:")
    print(
        df["sentiment_from_rating"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# 4. Main entry point
# --------------------------------------------------

if __name__ == "__main__":
    download_file()
    extract_reviews()
