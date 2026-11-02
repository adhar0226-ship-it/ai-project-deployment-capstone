import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_customer_data(df: pd.DataFrame):
    """
    Cleans raw customer data and handles feature engineering.
    """
    df = df.dropna().copy()
    features = df[['Annual_Income_k', 'Spending_Score']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler
