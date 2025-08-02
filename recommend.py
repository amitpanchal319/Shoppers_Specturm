import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ------------------------ Page Configuration ------------------------
st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")

# ------------------------ Load Data ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_sales_data.csv")

df = load_data()

# ------------------------ Load Models ------------------------
with open("models/svd_recommender.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open("models/rfm_scaler.pkl", "rb") as f:
    rfm_scaler = pickle.load(f)

with open("models/rfm_kmeans_model.pkl", "rb") as f:
    rfm_kmeans = pickle.load(f)

# ------------------------ UI Styling ------------------------
st.markdown("<h1 style='text-align: center; color: #2e7bcf;'>🛍️ Shopper Spectrum</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔁 Product Recommender", "🎯 Customer Segmentation"])

# ------------------------ Recommender System ------------------------
with tab1:
    st.subheader("🔍 Item-based Recommendation System")

    product_list = sorted(df['Description'].dropna().unique())
    selected_product = st.selectbox("Select a product:", product_list)

    if st.button("Get Recommendations"):
        pivot_table = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', fill_value=0)
        similarity_matrix = cosine_similarity(pivot_table.T)
        similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)

        if selected_product in similarity_df.columns:
            similar_items = similarity_df[selected_product].sort_values(ascending=False)[1:6]
            st.success("📌 Top 5 Recommended Products:")
            for i, (item, score) in enumerate(similar_items.items(), 1):
                st.write(f"{i}. {item} (Similarity: {score:.2f})")
        else:
            st.warning("Selected product not found in pivot table.")

# ------------------------ Customer Segmentation ------------------------
with tab2:
    st.subheader("🎯 RFM-Based Customer Segmentation")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("📅 Recency (days)", min_value=0)
    with col2:
        frequency = st.number_input("🔁 Frequency", min_value=0)
    with col3:
        monetary = st.number_input("💰 Monetary Value", min_value=0.0)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        input_scaled = rfm_scaler.transform(input_data)
        cluster = rfm_kmeans.predict(input_scaled)[0]

        labels = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }
        label = labels.get(cluster, f"Cluster {cluster}")
        st.success(f"🎯 Predicted Segment: **{label}** (Cluster {cluster})")

# ------------------------ Dashboard Visualization ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📊 Sales Dashboard")

col1, col2 = st.columns(2)

with col1:
    country_sales = df.groupby("Country")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig1 = px.bar(x=country_sales.index, y=country_sales.values,
                  labels={'x': 'Country', 'y': 'Total Quantity Sold'},
                  title="Top 10 Countries by Quantity Sold")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig2 = px.pie(values=top_products.values, names=top_products.index, title="Top 10 Selling Products")
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Shopper Spectrum | Streamlit App by Amit Panchal</p>", unsafe_allow_html=True)
