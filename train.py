import numpy as np
from model import build_lstm_model
from data_preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split

def train_model(data_path, feature_col='Close', sequence_length=60, epochs=50, batch_size=32):
    df = load_data(data_path)
    X, y, scaler = preprocess_data(df, feature_col, sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((sequence_length, 1))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, scaler, history
