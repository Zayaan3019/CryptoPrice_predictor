import numpy as np
from data_preprocessing import preprocess_data
from tensorflow.keras.models import load_model

def predict_next_price(model_path, recent_data, scaler):
    """
    Predict the next cryptocurrency price based on recent data.
    """
    model = load_model(model_path)
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    prediction = model.predict(recent_data_scaled.reshape(1, -1, 1))
    return scaler.inverse_transform(prediction)[0][0]
