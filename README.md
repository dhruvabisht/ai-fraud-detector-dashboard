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

## Author

**Dhruva Bisht**  







