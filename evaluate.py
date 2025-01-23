from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(true_values, predictions):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    print(f"MAE: {mae}, MSE: {mse}")
    return mae, mse
