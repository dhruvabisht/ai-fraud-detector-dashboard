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

st.set_page_config(page_title="Transaction Anomaly Detector", layout="wide")
st.title("AI-Powered Transaction Anomaly Detector")

# 2) File uploader
uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")
if not uploaded_file:
    st.warning("Please upload a CSV with columns: Amount, Latitude, Longitude, Device_Type, Transaction_Type, Timestamp")
    st.stop()

# 3) Read raw data
raw_df = pd.read_csv(uploaded_file)

# 4) Show expected schema vs. your upload
st.subheader("Expected schema")
st.table(pd.DataFrame({
    "Column": ["Amount","Latitude","Longitude","Device_Type","Transaction_Type","Timestamp"],
    "Type":   ["float", "float",   "float",   "string",      "string",            "datetime"]
}))

st.subheader("Your data sample")
st.dataframe(raw_df.head())

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
