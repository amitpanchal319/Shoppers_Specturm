# 🛒 Shopper Spectrum – Customer Segmentation & Product Recommendation System

Welcome to **Shopper Spectrum**, an intelligent Streamlit-based web app designed to segment customers using RFM (Recency, Frequency, Monetary) analysis and recommend products using collaborative filtering.

This tool empowers businesses to gain deeper customer insights and personalize their marketing and sales strategies.

---

## 🚀 Features

✅ **Customer Segmentation**  
- RFM-based customer profiling  
- KMeans clustering for segmentation  
- Scaled clustering model saved via `pickle`

✅ **Product Recommendation**  
- Collaborative filtering using SVD  
- Recommends personalized products based on past purchase behavior

✅ **Interactive Dashboard**  
- Streamlit-powered UI  
- Dropdown filters for user selection  
- Visualization of clusters and key metrics

✅ **Model Pipelines**  
- Models are pre-trained and stored as `.pkl` files:
  - `svd_recommender.pkl`
  - `rfm_scaler.pkl`
  - `rfm_kmeans_model.pkl`

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone 
cd ShopperSpectrum
2. Install Dependencies
pip install -r requirements.txt
3. Run the App

streamlit run recommend.py
📊 Tech Stack
Python

Pandas, NumPy, Scikit-learn

Streamlit

Pickle (for model serialization)

Matplotlib, Seaborn (for visualizations)

📸 Sample Visuals
(Add your app screenshots here if available)
Example:

Customer segments by cluster

Top recommendations per user

Sales trends and metrics

📦 Model Files
Make sure the following .pkl files are placed in the model/ directory:

rfm_kmeans_model.pkl

rfm_scaler.pkl

svd_recommender.pkl

And the cleaned data file is present in the data/ directory as:

cleaned_sales_data.csv

Link your GitHub repo and set recommend.py as the entry point.

**Amit Panchal**  
[📧 LinkedIn Profile](https://www.linkedin.com/in/amit-panchal0319/)


