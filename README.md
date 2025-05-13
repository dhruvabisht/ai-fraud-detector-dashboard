# AI-Powered Fraud Detection Dashboard

This project is a practical implementation of a real-time fraud detection system tailored for digital payments. It combines machine learning with a clean user interface to allow users to upload transaction data, detect anomalies, and visualize potential fraud cases instantly.

## Built using:
- **Python** for ML logic and data processing
- **Streamlit** for the interactive dashboard UI
- **Matplotlib/Seaborn** for analytics visualizations

## Designed to reflect real-world use cases in the financial services industry, especially in roles involving:
- Product Management for AI Solutions
- Payments Analytics
- Risk/Fraud Detection Systems

## Features

- Machine learning-based fraud detection
- Upload and analyze transaction datasets
- Visual insights into fraud patterns
- Real-time risk prediction
- Deployable via Streamlit Cloud (free hosting)

## How It Works

- Trained a RandomForestClassifier on synthetic transaction data
- Accepts new transaction CSVs via upload
- Predicts fraud likelihood per transaction
- Displays visual and textual summaries of results

## Project Structure
fraud-detector-ai/
├── app/
│ └── dashboard.py # Main Streamlit app
├── models/
│ └── fraud_detector.py # ML model logic
├── utils/
│ └── helper.py # Utility functions
├── data/
│ └── sample.csv # Sample data
├── requirements.txt # Python dependencies
└── README.md

## Run Locally

1. Clone this repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the app:  
   `streamlit run app/dashboard.py`

## Live Demo

✅ [Live Streamlit App](https://your-username.streamlit.app)  
(Replace this link after deployment)

## Screenshots

Upload screenshots here after hosting (e.g., dashboard overview, fraud alert panel)

## Use Case

Designed to show how AI can be used in fintech to proactively monitor and prevent fraud in real-time. This project is part of a career portfolio focused on applied machine learning and product thinking in financial technology.

## Acknowledgements

- Streamlit (for rapid dashboard creation)
- Scikit-learn (for the ML backend)
- Inspired by fraud detection use cases in digital payments

## Author

**Dhruva Bisht**  







