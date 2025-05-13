import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class FraudDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.02)
        self.scaler = StandardScaler()

    def preprocess(self, df):
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        X = df[['Amount', 'Hour', 'DayOfWeek']]
        return self.scaler.fit_transform(X)

    def fit(self, df):
        X = self.preprocess(df)
        self.model.fit(X)

    def predict(self, df):
        X = self.preprocess(df)
        preds = self.model.predict(X)
        df['Anomaly'] = preds
        df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})
        return df