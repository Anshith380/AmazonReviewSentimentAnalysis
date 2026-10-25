
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amazon_electronics_reviews_cleaned.csv"
)

st.set_page_config(
    page_title="Electronics Product Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Amazon Electronics Product Intelligence")
st.caption("Product ratings, customer sentiment, and review analytics")

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()


@st.cache_data
def load_reviews(path):
    return pd.read_csv(path)


df = load_reviews(str(DATA_FILE))

required = {
    "product_id",
    "rating",
    "review_text",
    "sentiment_from_rating",
}
if not required.issubset(df.columns):
    st.error("The dataset is missing required columns.")
    st.stop()

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(
    subset=["product_id", "rating", "review_text"]
).copy()

st.sidebar.header("Filters")

products = sorted(df["product_id"].astype(str).unique())
selected_products = st.sidebar.multiselect(
    "Products",
    options=products,
    default=[],
    help="Leave empty to include all products.",
)

sentiments = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=sentiments,
    default=sentiments,
)

min_rating, max_rating = st.sidebar.select_slider(
    "Star rating range",
    options=[1, 2, 3, 4, 5],
    value=(1, 5),
)

filtered = df[
    df["rating"].between(min_rating, max_rating)
    & df["sentiment_from_rating"].isin(selected_sentiments)
].copy()

if selected_products:
    filtered = filtered[
        filtered["product_id"].astype(str).isin(selected_products)
    ]

search = st.sidebar.text_input("Search review text")
if search.strip():
    filtered = filtered[
        filtered["review_text"].str.contains(
            search.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]

total = len(filtered)
avg_rating = filtered["rating"].mean() if total else 0
product_count = filtered["product_id"].nunique()
positive_pct = (
    100 * (filtered["sentiment_from_rating"] == "Positive").mean()
    if total else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{total:,}")
c2.metric("Average rating", f"{avg_rating:.2f} / 5")
c3.metric("Products", f"{product_count:,}")
c4.metric("Positive share", f"{positive_pct:.1f}%")

st.divider()

tab_overview, tab_products, tab_reviews = st.tabs([
    "Overview",
    "Product Comparison",
    "Review Explorer",
])

with tab_overview:
    st.subheader("Star Rating Distribution")

    if filtered.empty:
        st.info("No reviews match the selected filters.")
    else:
        rating_counts = (
            filtered["rating"]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("Stars")
            .reset_index(name="Reviews")
        )

        fig = px.bar(
            rating_counts,
            x="Stars",
            y="Reviews",
            title="Customer ratings",
            text="Reviews",
        )
        st.plotly_chart(fig, use_container_width=True)

        sentiment_counts = (
            filtered["sentiment_from_rating"]
            .value_counts()
            .reindex(sentiments, fill_value=0)
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            hole=0.4,
            title="Rating-derived sentiment distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_products:
    st.subheader("Compare Products")

    if filtered.empty:
        st.info("No products match the selected filters.")
    else:
        product_summary = (
            filtered.groupby("product_id")
            .agg(
                review_count=("rating", "size"),
                average_rating=("rating", "mean"),
                positive_reviews=(
                    "sentiment_from_rating",
                    lambda values: (values == "Positive").sum(),
                ),
                negative_reviews=(
                    "sentiment_from_rating",
                    lambda values: (values == "Negative").sum(),
                ),
            )
            .reset_index()
        )

        product_summary["average_rating"] = (
            product_summary["average_rating"].round(2)
        )

        min_reviews = st.slider(
            "Minimum reviews per product",
            min_value=1,
            max_value=20,
            value=3,
        )

        product_summary = product_summary[
            product_summary["review_count"] >= min_reviews
        ]

        sort_by = st.selectbox(
            "Rank products by",
            [
                "average_rating",
                "review_count",
                "positive_reviews",
                "negative_reviews",
            ],
        )

        product_summary = product_summary.sort_values(
            sort_by, ascending=False
        )

        st.dataframe(
            product_summary,
            use_container_width=True,
            hide_index=True,
        )

        if not product_summary.empty:
            fig = px.scatter(
                product_summary,
                x="review_count",
                y="average_rating",
                size="review_count",
                hover_name="product_id",
                hover_data=[
                    "positive_reviews",
                    "negative_reviews",
                ],
                title="Product ratings vs. review volume",
                labels={
                    "review_count": "Number of reviews",
                    "average_rating": "Average star rating",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Product Comparison CSV",
                data=product_summary.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="product_comparison.csv",
                mime="text/csv",
            )

with tab_reviews:
    st.subheader("Customer Review Explorer")

    display_columns = [
        col for col in [
            "product_id",
            "rating",
            "sentiment_from_rating",
            "review_title",
            "review_text",
            "review_date",
            "helpful_votes",
        ]
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Reviews CSV",
        data=filtered.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="filtered_electronics_reviews.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Dataset: historical Amazon Electronics reviews. "
    "Sentiment is derived from star ratings: 1–2 negative, "
    "3 neutral, and 4–5 positive. Product IDs identify items "
    "but do not provide product names. Results describe this "
    "sample, not all current Amazon customers."
)
