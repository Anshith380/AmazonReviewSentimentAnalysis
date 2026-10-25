# Amazon Review Intelligence

An NLP and machine-learning application that analyzes Amazon customer reviews, classifies sentiment, and presents review insights through an interactive Streamlit dashboard.

## Features

- Collection of 10,000 reviews from a public Amazon sentiment dataset.
- Text cleaning and preprocessing.
- TF-IDF feature extraction.
- Logistic Regression sentiment classification.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Interactive Streamlit dashboard.
- Review search and CSV export.
- Automated preprocessing tests.

## Technology Stack

Python, Pandas, scikit-learn, NLTK, Streamlit, Plotly, Joblib, Pytest.

## Project Structure

```text
AmazonReviewSentimentAnalysis/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── download_reviews.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── export_reviews.py
├── tests/
│   └── test_preprocessing.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── reports/
```

## Setup

1. Install Python 3.12 and Git.
2. Clone the repository.
3. Create and activate a virtual environment.
4. Install the dependencies.
5. Obtain the dataset and prepare the cleaned CSV.
6. Train the model.
7. Start the dashboard.

### Windows commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\preprocessing.py
python .\src\train_model.py
python -m streamlit run app.py
```

The raw dataset must be available at `data/raw/amazon_reviews_raw.csv` before preprocessing. Generated datasets and model artifacts are excluded from Git.

## Model Evaluation

The initial TF-IDF and Logistic Regression model achieved approximately 86% test accuracy on a held-out set of 2,000 reviews. Refer to the generated reports for the complete evaluation metrics.

## Dataset and Limitations

The project uses the public `fancyzhx/amazon_polarity` dataset. The initial sample contains 10,000 reviews with positive and negative source labels. These labels do not provide original star ratings or reliable product-level identifiers for comparisons.

The reported metrics measure agreement with the source sentiment labels, not guaranteed real-world sentiment understanding.

## Future Improvements

- Evaluate additional classification algorithms.
- Add more comprehensive data-quality checks.
- Obtain product-level metadata for valid product comparisons.
- Improve error handling and dashboard testing.
- Deploy through an approved hosting environment.

## License

Check the dataset's license and terms before redistributing the data. Add an appropriate license for your own code before publishing.
