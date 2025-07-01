import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def build_preprocessor():
    """
    Create a preprocessing pipeline that:
      - Imputes numeric missing values with median, then scales.
      - Imputes categorical missing values with 'Unknown', then one-hot encodes.
      - Parses timestamp column into hour and day-of-week features.
    Returns:
        preprocessor (ColumnTransformer): the assembled transformer.
    """
    # Define your expected columns
    num_cols = ["Amount", "Latitude", "Longitude"]
    cat_cols = ["Device_Type", "Transaction_Type"]
    time_col = ["Timestamp"]

    # Numeric pipeline: median imputation + standard scaling
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    # Categorical pipeline: constant imputation + one-hot encoding
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    # Timestamp pipeline: parse to datetime, extract hour and day-of-week
    def extract_time_features(df):
        df = pd.to_datetime(df["Timestamp"], errors="coerce")
        return pd.DataFrame({
            "hour": df.dt.hour.fillna(-1).astype(int),
            "dayofweek": df.dt.dayofweek.fillna(-1).astype(int)
        })

    time_pipe = Pipeline([
        ("extract", FunctionTransformer(extract_time_features, validate=False))
    ])

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("time", time_pipe, time_col)
        ],
        remainder="drop"
    )

    return preprocessor


class DataPipeline:
    """
    Wrapper for preprocessing raw DataFrames for fraud detection.
    Usage:
        dp = DataPipeline()
        X_clean = dp.fit_transform(df_raw)
        X_new = dp.transform(df_new)
    """
    def __init__(self):
        self.preprocessor = build_preprocessor()
        self.fitted = False

    def _validate_schema(self, df: pd.DataFrame):
        """
        Ensure that raw DataFrame contains all required columns.
        """
        expected = set(
            ["Amount", "Latitude", "Longitude", "Device_Type", "Transaction_Type", "Timestamp"]
        )
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def fit(self, df: pd.DataFrame):
        """
        Learn any statistics from training data (e.g. medians, categories).
        """
        self._validate_schema(df)
        # Fit the preprocessor to the DataFrame
        self.preprocessor.fit(df)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """
        Apply the fitted preprocessing to new data.
        """
        if not self.fitted:
            raise RuntimeError("DataPipeline must be fitted before calling transform.")
        self._validate_schema(df)
        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        """
        Shortcut for fit followed by transform.
        """
        self.fit(df)
        return self.transform(df)


if __name__ == "__main__":
    # Quick smoke test when running as a script
    sample = pd.DataFrame({
        "Amount": [100.0, None, 250.5],
        "Latitude": [52.3, 53.1, None],
        "Longitude": [-6.4, None, -7.8],
        "Device_Type": ["Mobile", None, "Desktop"],
        "Transaction_Type": ["Online", "In-store", None],
        "Timestamp": ["2025-04-16T04:24:35", "2025-05-01T12:00:00", None]
    })
    dp = DataPipeline()
    X = dp.fit_transform(sample)
    print("Cleaned shape:", X.shape)
