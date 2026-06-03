import numpy as np
import joblib
import torch
import torch.nn as nn
from random import randint

# ==========================
# Load Model and Scaler
# ==========================

class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = TrafficLSTM()
model.load_state_dict(torch.load("traffic_lstm_model.pt"))
model.eval()

scaler = joblib.load("traffic_scaler.save")


# ==========================
# Prediction Function
# ==========================

def predict_next_hour(vehicle_list):
    """
    vehicle_list : list of last 10 vehicle counts
    returns : predicted next hour vehicle count (integer)
    """
    if len(vehicle_list) != 10:
        return randint(2, 10)
    
    # Convert to numpy array
    arr = np.array(vehicle_list, dtype=np.float32).reshape(-1, 1)
    
    # Scale input
    scaled_input = scaler.transform(arr)
    
    # Reshape for LSTM: [batch=1, seq_len=10, features=1]
    scaled_input = scaled_input.reshape(1, 10, 1)
    
    # Run PyTorch inference
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        prediction_tensor = model(input_tensor)
        prediction = prediction_tensor.numpy()
    
    # Inverse scale
    predicted_value = scaler.inverse_transform(prediction)
    
    return int(round(predicted_value[0][0]))


# ==========================
# Example Usage
# ==========================

if __name__ == "__main__":
    sample_input = [5, 6, 7, 8, 10, 12, 14, 15, 13, 11]
    result = predict_next_hour(sample_input)
    print("Input:", sample_input)
    print("Predicted Next Hour Vehicles:", result)