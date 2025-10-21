# app/streamlit_app.py

import os
import sys

# 1) Ensure Python can see both app/ and the top-level models/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from data_pipeline import DataPipeline
from models.fraud_detector import FraudDetector
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI-Powered Transaction Anomaly Detector – Demo Mode Available", layout="wide")
st.title("AI-Powered Transaction Anomaly Detector – Demo Mode Available")

# 2) File uploader with demo fallback
uploaded_file = st.file_uploader("Upload Transaction CSV (or use demo dataset below)", type="csv")

# 3) Read raw data - auto-load demo if no file uploaded
if uploaded_file is None:
    st.info("📊 Demo dataset loaded for preview. Upload your own CSV above to analyze your data.")
    try:
        raw_df = pd.read_csv("synthetic_transactions.csv")
    except FileNotFoundError:
        st.error("Demo dataset not found. Please upload a CSV file.")
        st.stop()
else:
    raw_df = pd.read_csv(uploaded_file)

# 4) Show expected schema vs. your upload
st.subheader("Expected schema")
st.table(pd.DataFrame({
    "Column": ["Amount","Latitude","Longitude","Device_Type","Transaction_Type","Timestamp"],
    "Type":   ["float", "float",   "float",   "string",      "string",            "datetime"]
}))

st.subheader("Your data sample")
st.dataframe(raw_df.head())

# Demo insights card
if uploaded_file is None:
    st.subheader("📈 Sample Insights (Demo)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(raw_df):,}")
    with col2:
        st.metric("Expected Anomalies", "~100 (2%)")
    with col3:
        st.metric("Data Quality", "✅ Clean")
    
    st.info("💡 This demo shows how your transaction data would be analyzed. Upload your own CSV to see real fraud detection results!")

# 5) Clean & transform
dp = DataPipeline()
try:
    X_clean = dp.fit_transform(raw_df)
except Exception as e:
    st.error(f"Data validation/cleaning failed:\n>{e}")
    st.stop()
else:
    st.success("✅ Data cleaned and validated!")

# 6) Fit & predict
fd = FraudDetector()
# If your model requires a separate train/test or pre-trained weights, adjust accordingly
fd.fit(raw_df)  
result = fd.predict(raw_df)

# 7) Surface metrics
frauds = result[result["Anomaly"] == 1]
st.metric("Total Transactions", len(result))
st.metric("Detected Anomalies",    len(frauds))

# 8) Show results
st.subheader("Sample Transactions with Scores")
st.dataframe(result.head(20))

# 9) Fraud by hour chart
st.subheader("Fraud Detection by Hour")
fig, ax = plt.subplots()
sns.histplot(
    result[result["Anomaly"] == 1]["Timestamp"]
      .apply(pd.to_datetime)
      .dt.hour,
    bins=24, ax=ax
)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Count of Anomalies")
st.pyplot(fig)
