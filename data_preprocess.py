import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Load cryptocurrency data from a CSV file.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def preprocess_data(df, feature_col='Close', sequence_length=60):
    """
    Preprocess data for time-series forecasting.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_col]])
    sequences, targets = [], []
    for i in range(sequence_length, len(scaled_data)):
        sequences.append(scaled_data[i-sequence_length:i, 0])
        targets.append(scaled_data[i, 0])
    return np.array(sequences), np.array(targets), scaler
