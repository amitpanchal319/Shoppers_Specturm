# 🛒 Shopper Spectrum – Customer Segmentation & Product Recommendation System
Welcome to Shopper Spectrum, a powerful and intuitive web application built with Streamlit that performs Customer Segmentation using RFM (Recency, Frequency, Monetary) analysis and provides Product Recommendations using Collaborative Filtering (SVD).

This tool helps businesses understand customer behavior, optimize marketing strategies, and enhance user experience through personalized product suggestions.

# 🧠 Problem Statement
The global e-commerce industry produces enormous transaction datasets daily, which, when analyzed, can unlock meaningful patterns in consumer behavior. This project focuses on:

Segmenting customers based on RFM analysis

Applying KMeans clustering to identify distinct customer groups

Implementing a recommendation system using Singular Value Decomposition (SVD) to suggest products based on past purchases

# 🚀 Features
✅ Customer Segmentation
RFM (Recency, Frequency, Monetary) based profiling

Scaled data using MinMaxScaler

KMeans clustering for identifying customer segments

Pre-trained model loaded from rfm_kmeans_model.pkl

✅ Product Recommendation
Personalized recommendations using Collaborative Filtering (SVD)

Learns from past purchase behavior to recommend top N products

Model loaded from svd_recommender.pkl

✅ Interactive Streamlit Dashboard
Clean and modern UI

Dropdown filters for selecting customer or product

Real-time product recommendations

Cluster visualization and summary stats

✅ Efficient Model Pipelines
Pre-trained models are saved as .pkl files and loaded at runtime to improve performance:


📦 models/
├── rfm_scaler.pkl
├── rfm_kmeans_model.pkl
└── svd_recommender.pkl
📊 Visualizations
The dashboard includes:

📈 Cluster Distribution (Pie/Bar charts)

📊 RFM Score Analysis

🧮 Segment Summary Tables

🧠 Product Recommendation Preview

🛠️ Tech Stack
Frontend/UI: Streamlit

Data Analysis: Pandas, NumPy

Modeling: Scikit-learn, Surprise (SVD), Joblib/Pickle

Visualization: Matplotlib, Seaborn, Plotly

Amit Panchal
🔗 LinkedIn Profile