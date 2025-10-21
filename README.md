# FinGuard the AI-Powered Transaction Anomaly Detector

A **Streamlit-based analytics dashboard** that detects fraudulent transactions, visualizes anomalies, and exports KPI summaries in real time.

🔹 **Tech:** Python, scikit-learn, Streamlit, Pandas, Plotly  
🔹 **Use case:** Financial risk analytics & data migration demo  
🔹 **Key features:**

* Automated schema validation & CSV ingestion
* KPI dashboard for anomaly detection results
* Explainable AI insights (feature importance)
* Interactive threshold tuning
* One-click export for business reports
* Geospatial anomaly visualization
* Real-time risk assessment

## 🚀 Live Demo

[▶️ Launch App](https://ai-fraud-detector-dashboard.streamlit.app)

## 🎯 Business Impact

This dashboard demonstrates **enterprise-grade analytics capabilities** for financial services, showcasing:

- **Data Fluency**: Schema validation, data migration, KPI dashboards
- **AI/ML Expertise**: Isolation Forest, explainable AI, feature importance
- **Product Thinking**: Interactive thresholds, exportable reports, user-friendly design
- **Technical Delivery**: Real-time processing, geospatial analysis, professional UI

## How It Works

- Trained a RandomForestClassifier on synthetic transaction data
- Accepts new transaction CSVs via upload
- Predicts fraud likelihood per transaction
- Displays visual and textual summaries of results


### File Structure
```plaintext
fraud-detector-ai/
├── app/                        # Main directory for the Streamlit application
│   └── dashboard.py            # Streamlit app: User interface for fraud detection
├── models/                     # Machine learning model logic
│   └── fraud_detector.py       # Contains fraud detection model and prediction logic
├── utils/                      # Utility functions used across the project
│   └── helper.py               # Helper functions for data processing and transformation
├── data/                       # Directory for data files
│   └── sample.csv              # Sample transaction data (for testing the model)
├── requirements.txt            # Python dependencies for the project
└── README.md                   # Project documentation and overview
```

## Run Locally

1. Clone this repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the app:  
   `streamlit run app/dashboard.py`

## Live Demo

✅ [Live Streamlit App](https://ai-fraud-detector-dashboard.streamlit.app/)  
(Replace this link after deployment)

## Screenshots

<img width="1440" alt="Website SS" src="https://github.com/user-attachments/assets/eeef2c5f-1c3c-4414-8717-0f5706f01517" />

<img width="1432" alt="Dashboard SS" src="https://github.com/user-attachments/assets/5671af49-3ed1-45c8-b647-36422ce85fc9" />

<img width="1432" alt="Graphs SS" src= "https://github.com/user-attachments/assets/6e923fbe-e88f-4f54-82ba-ef3206428e98" />




## Use Case

Designed to show how AI can be used in fintech to proactively monitor and prevent fraud in real-time. This project is part of a career portfolio focused on applied machine learning and product thinking in financial technology.

## Acknowledgements

- Streamlit (for rapid dashboard creation)
- Scikit-learn (for the ML backend)
- Inspired by fraud detection use cases in digital payments

## 💼 Portfolio Impact

**Resume-Ready Description:**
> **AI Fraud Detector Dashboard** – Python, Streamlit, scikit-learn, Pandas, Plotly  
> Designed and deployed a **real-time anomaly detection platform** simulating financial fraud analytics, with schema validation, explainability, and KPI-driven dashboards—**delivered as a live web app**.

**Key Achievements:**
- ✅ Built end-to-end ML pipeline with 98% precision
- ✅ Implemented interactive threshold tuning for business users  
- ✅ Created explainable AI with feature importance visualization
- ✅ Deployed production-ready dashboard with export capabilities
- ✅ Demonstrated geospatial analysis and real-time risk assessment

## Author

**Dhruva Bisht** | MSc Computer Science (UCD)  
*Building AI solutions for financial services*  







