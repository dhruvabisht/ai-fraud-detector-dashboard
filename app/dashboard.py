import streamlit as st
import pandas as pd
from models.fraud_detector import FraudDetector
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Transaction Anomaly Detector", layout="wide")
st.title("💳 AI-Powered Transaction Anomaly Detector")

uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    fd = FraudDetector()
    fd.fit(df)
    result = fd.predict(df)

    frauds = result[result['Anomaly'] == 1]
    st.metric("Total Transactions", len(result))
    st.metric("Detected Anomalies", len(frauds))

    st.subheader("Sample Transactions")
    st.dataframe(result.head(20))

    st.subheader("Fraud Detection by Hour")
    fig, ax = plt.subplots()
    sns.histplot(result[result['Anomaly']==1]['Timestamp'].apply(pd.to_datetime).dt.hour, bins=24, ax=ax)
    st.pyplot(fig)