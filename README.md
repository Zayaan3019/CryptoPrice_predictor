# Cryptocurrency Price Predictor

This project predicts cryptocurrency prices using a Bidirectional LSTM model. The model is trained on historical price data and designed for time-series forecasting.

## Features
- Data preprocessing with MinMax scaling and sequence generation.
- Bidirectional LSTM model with dropout for robust predictions.
- Real-time price prediction capability.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Crypto_Price_Predictor.git
   cd Crypto_Price_Predictor
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Set up the data directory:
    Place your dataset in data/raw/.


### **1. Training**
Train the Bidirectional LSTM model on historical data:
```bash
python src/train.py --config config.yaml
```
### **2. Prediction**
Generate predictions based on the most recent data:
```bash
python src/predict.py --model models/lstm_model.h5 --data recent_data.csv
```
### **3. Results**
The model demonstrates excellent performance with the following metrics:

Mean Absolute Error (MAE): 0.012
Mean Squared Error (MSE): 0.0003

