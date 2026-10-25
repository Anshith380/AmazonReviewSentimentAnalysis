
from pathlib import Path
import json
import re

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "sentiment_model.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "amazon_reviews_cleaned.csv"
METRICS_FILE = PROJECT_ROOT / "reports" / "model_metrics.json"
CONFUSION_FILE = PROJECT_ROOT / "reports" / "confusion_matrix.csv"

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Amazon Review Intelligence")
st.caption("Customer feedback analytics and sentiment prediction")

if not MODEL_FILE.exists() or not DATA_FILE.exists():
    st.error("Model or cleaned dataset not found. Train the model first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

model = load_model(str(MODEL_FILE))
df = load_data(str(DATA_FILE))

# Sidebar filters
st.sidebar.header("Dashboard Filters")

if "sentiment" in df.columns:
    sentiments = sorted(df["sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=sentiments,
        default=sentiments,
    )
else:
    selected_sentiments = []

filtered = df[df["sentiment"].isin(selected_sentiments)].copy()

if "review_text" in filtered.columns:
    search = st.sidebar.text_input("Search reviews")
    if search.strip():
        filtered = filtered[
            filtered["review_text"].str.contains(
                search.strip(), case=False, na=False, regex=False
            )
        ]

# KPI metrics
total_reviews = len(filtered)
positive_count = int((filtered["sentiment"] == "Positive").sum())
negative_count = int((filtered["sentiment"] == "Negative").sum())
positive_pct = (
    100 * positive_count / total_reviews if total_reviews else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{total_reviews:,}")
c2.metric("Positive", f"{positive_count:,}")
c3.metric("Negative", f"{negative_count:,}")
c4.metric("Positive share", f"{positive_pct:.1f}%")

st.divider()

tab_overview, tab_predict, tab_data, tab_model = st.tabs([
    "Overview",
    "Predict Sentiment",
    "Explore Reviews",
    "Model Performance",
])

with tab_overview:
    st.subheader("Sentiment Distribution")

    if not filtered.empty:
        counts = (
            filtered["sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        fig = px.pie(
            counts,
            names="Sentiment",
            values="Reviews",
            hole=0.45,
            title="Distribution of source sentiment labels",
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Most Frequent Words")
            text_column = (
                "cleaned_text"
                if "cleaned_text" in filtered.columns
                else "review_text"
            )
            words = re.findall(
                r"\b[a-zA-Z]{3,}\b",
                " ".join(filtered[text_column].astype(str).head(2000)),
            )
            stop_words = {
                "the", "and", "for", "that", "this", "with",
                "was", "are", "but", "have", "you", "not",
                "they", "its", "from", "very", "just", "one",
                "all", "out", "has", "had", "too", "when",
            }
            word_counts = pd.Series(
                [word for word in words if word not in stop_words]
            ).value_counts().head(15)

            if not word_counts.empty:
                word_fig = px.bar(
                    x=word_counts.values,
                    y=word_counts.index,
                    orientation="h",
                    labels={"x": "Frequency", "y": "Word"},
                    title="Frequent terms in sampled reviews",
                )
                st.plotly_chart(word_fig, use_container_width=True)

        with right:
            st.subheader("Source Label Counts")
            bar = px.bar(
                counts,
                x="Sentiment",
                y="Reviews",
                color="Sentiment",
                title="Reviews by sentiment",
            )
            st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No reviews match the current filters.")

with tab_predict:
    st.subheader("Analyze a Customer Review")
    review_input = st.text_area(
        "Enter review text",
        placeholder="Example: Excellent quality and great value for money!",
        height=150,
    )

    if st.button("Analyze Review", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a review first.")
        else:
            prediction = model.predict([review_input])[0]
            probabilities = model.predict_proba([review_input])[0]
            classes = model.classes_
            confidence = float(probabilities.max())

            if prediction == "Positive":
                st.success(f"Predicted sentiment: {prediction}")
            else:
                st.error(f"Predicted sentiment: {prediction}")

            st.metric("Model confidence", f"{confidence:.1%}")

            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities,
            })
            probability_fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                range_y=[0, 1],
                title="Prediction probabilities",
            )
            st.plotly_chart(probability_fig, use_container_width=True)

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

with tab_data:
    st.subheader("Review Explorer")
    st.write(f"Showing {len(filtered):,} reviews after filtering.")

    display_columns = [
        col for col in [
            "review_id",
            "review_title",
            "review_text",
            "cleaned_text",
            "sentiment",
        ] if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].head(500),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Filtered Reviews (CSV)",
        data=csv_bytes,
        file_name="filtered_amazon_reviews.csv",
        mime="text/csv",
    )

with tab_model:
    st.subheader("Model Evaluation")

    if METRICS_FILE.exists():
        metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        st.write(f"Training reviews: {metrics['training_rows']:,}")
        st.write(f"Testing reviews: {metrics['testing_rows']:,}")
        st.text(metrics["classification_report"])

    if CONFUSION_FILE.exists():
        matrix = pd.read_csv(CONFUSION_FILE, index_col=0)
        st.write("Confusion Matrix")
        st.dataframe(matrix, use_container_width=True)

st.divider()
st.caption(
    "Data source: public Amazon review sentiment dataset. "
    "Labels originate from the dataset; the displayed analytics "
    "describe this dataset, not all current Amazon customers."
)
