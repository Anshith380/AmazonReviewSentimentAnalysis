# Amazon Review Intelligence

An end-to-end Natural Language Processing (NLP) and machine-learning project that analyzes Amazon customer reviews, predicts sentiment, explores product ratings, and presents insights through interactive Streamlit dashboards.

## Project Overview

This project provides two complementary workflows:

1. **Review Sentiment Classification:** Uses TF-IDF feature extraction and Logistic Regression to classify review text as positive or negative.
2. **Product Review Analytics:** Analyzes Amazon Electronics reviews using product identifiers, star ratings, derived sentiment labels, and interactive visualizations.

The goal is to turn customer feedback into useful insights that can help understand product satisfaction, identify negative experiences, and explore customer opinions.

## Key Features

- Download and prepare Amazon review datasets.
- Clean and preprocess review text.
- Convert text into numerical TF-IDF features.
- Train a Logistic Regression sentiment classifier.
- Evaluate model performance using classification metrics and a confusion matrix.
- Explore product-level ratings and sentiment distributions.
- Filter reviews by product, rating, and sentiment.
- Compare products using available review data.
- Search and inspect individual reviews.
- Export review data as CSV files.
- Validate preprocessing and product-review data using automated tests.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| Pandas and NumPy | Data manipulation and analysis |
| scikit-learn | TF-IDF, Logistic Regression, and model evaluation |
| NLTK | Text-processing utilities |
| Streamlit | Interactive web dashboards |
| Plotly, Matplotlib, Seaborn | Data visualization |
| Joblib | Saving and loading trained models |
| Pytest | Automated testing |
| Hugging Face Datasets | Accessing the initial sentiment dataset |
| Requests | Downloading public review data |

## Project Structure

```text
AmazonReviewSentimentAnalysis/
├── app.py
├── product_dashboard.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── download_reviews.py
│   ├── download_product_reviews.py
│   ├── export_reviews.py
│   ├── preprocessing.py
│   ├── preprocess_product_reviews.py
│   ├── sentiment.py
│   ├── train_model.py
│   └── visualization.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_product_reviews.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── reports/
```

Generated datasets, trained model files, and reports are stored locally as needed. Their availability depends on which workflows you have run.

## Dataset Information

### 1. Initial sentiment-classification dataset

The initial workflow uses the public `fancyzhx/amazon_polarity` dataset accessed through Hugging Face Datasets.

- Sample size: 10,000 reviews.
- Source labels: positive and negative.
- Main purpose: train and evaluate a binary sentiment classifier.
- The original source labels are used as the target labels for model evaluation.

### 2. Electronics product-review dataset

The product analytics workflow uses the historical Amazon Electronics 5-core review data published by Stanford's SNAP project.

Dataset source:

https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

The project extracts 10,000 valid review records from the downloaded source and removes duplicate records during preparation.

The resulting dataset contains 9,999 unique review rows and 407 distinct product identifiers in the current prepared sample.

The available fields include product ID, rating, review title, review text, review date, reviewer ID, and helpful-vote information where available.

**Important dataset limitations:**

- The sample consists of the first 10,000 valid records extracted from the source, not a random or representative sample of all Electronics reviews.
- It reflects historical reviews, not current Amazon product ratings.
- Product sentiment labels are derived from star ratings: ratings of 4–5 are Positive, 3 is Neutral, and 1–2 are Negative.
- These derived labels are rating-based categories, not independent human annotations of the review text.

Check the dataset provider's terms and license before redistributing the dataset.

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Visual Studio Code or another Python editor
- Internet access for downloading dependencies and datasets

### 1. Clone the repository

```powershell
git clone https://github.com/Anshith380/AmazonReviewSentimentAnalysis.git
cd AmazonReviewSentimentAnalysis
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell cannot activate the virtual environment, consult the Python and PowerShell documentation or use an approved terminal configuration.

## Workflow A: Binary Sentiment Classification

This workflow trains a text classifier using the initial sentiment dataset.

### Step 1: Prepare the dataset

The raw CSV is expected at:

```text
data/raw/amazon_reviews_raw.csv
```

Obtain the dataset using the repository's data-download workflow before running preprocessing.

### Step 2: Preprocess the text

```powershell
python .\src\preprocessing.py
```

This prepares the review text for machine learning and generates the processed dataset.

### Step 3: Train the model

```powershell
python .\src\train_model.py
```

The training workflow uses TF-IDF features and Logistic Regression. It evaluates the model on held-out review data and saves model artifacts and evaluation outputs locally.

### Step 4: Launch the original dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Workflow B: Electronics Product Analytics

### Step 1: Download the Electronics reviews

```powershell
python .\src\download_product_reviews.py
```

This script downloads and processes a sample from the public historical Electronics review dataset.

The raw review data is saved under:

```text
data/raw/amazon_electronics_reviews.csv
```

The download requires an internet connection and sufficient disk space for the compressed source dataset.

### Step 2: Preprocess the product reviews

```powershell
python .\src\preprocess_product_reviews.py
```

The preprocessing workflow cleans the review text and prepares the product-level dataset.

Expected processed file:

```text
data/processed/amazon_electronics_reviews_cleaned.csv
```

### Step 3: Launch the product analytics dashboard

```powershell
python -m streamlit run product_dashboard.py
```

Open the local URL displayed in the terminal.

### Dashboard capabilities

The product dashboard supports exploration of:

- Review counts and average star ratings.
- Product-level review coverage.
- Rating-derived sentiment distributions.
- Review filtering and inspection.
- Product comparisons based on the available sample.
- CSV exports for further analysis.

The displayed statistics change according to the active filters. They should not be interpreted as current Amazon-wide statistics.

## Model Evaluation

The initial TF-IDF and Logistic Regression classifier achieved approximately **86% accuracy** on a held-out test set of 2,000 reviews in the recorded run.

Evaluation includes accuracy, precision, recall, F1-score, and a confusion matrix.

Accuracy alone does not fully describe classifier quality. Review the per-class precision, recall, and F1-score when assessing model performance.

The recorded result applies to the initial binary classification workflow and should not be presented as the performance of the rating-derived Electronics sentiment categories.

Re-run training to reproduce the metrics in your current environment.

## Automated Testing

The project includes tests for text preprocessing and product-review data validation.

Run all tests from the repository root:

```powershell
python -m pytest -v
```

The current recorded test run completed with **14 passed tests**.

Coverage includes:

- Lowercase conversion.
- HTML and URL removal.
- Whitespace normalization.
- Preservation of negation.
- Missing and empty text handling.
- Required product-review columns.
- Valid star-rating ranges.
- Rating-to-sentiment mapping.
- Product IDs and non-empty review fields.

These tests validate the implemented checks; they do not replace end-to-end testing of every dashboard feature or the full model-training workflow.

## Business Value

Customer reviews provide valuable feedback about product quality, usability, satisfaction, and common problems.

This project demonstrates how NLP, machine learning, and interactive analytics can help:

- Summarize positive and negative customer feedback.
- Explore rating distributions across products.
- Identify products with more negative reviews in the available sample.
- Support data-driven investigation of customer complaints.
- Prepare structured review data for reporting and further analysis.

The results are exploratory and should be validated against larger, more representative datasets before being used for business decisions.

## Limitations and Future Improvements

- Use representative and larger review samples.
- Add end-to-end integration tests for data preparation, training, and dashboards.
- Compare Logistic Regression with additional classification algorithms.
- Improve model evaluation on imbalanced data.
- Add