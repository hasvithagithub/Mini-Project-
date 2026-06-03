import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# 1. Define PyTorch LSTM Model
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Out shape: [batch, seq_len, hidden_size]. We take last time step.
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model():
    print("Loading dataset...")
    df = pd.read_csv("traffic_dataset.csv")
    data = df["number_of_vehicles"].values.reshape(-1, 1).astype(np.float32)
    
    print("Fitting MinMaxScaler...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler (which is expected by predicter.py)
    joblib.dump(scaler, "traffic_scaler.save")
    print("Scaler saved to traffic_scaler.save")
    
    # Create sequences
    seq_length = 10
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])
        y.append(scaled_data[i + seq_length])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
    
    # Create Model, Loss, Optimizer
    model = TrafficLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    batch_size = 32
    
    print("Training PyTorch LSTM Model...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(X_train_t)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    # Save Model Weights
    torch.save(model.state_dict(), "traffic_lstm_model.pt")
    print("Model saved to traffic_lstm_model.pt")

if __name__ == "__main__":
    train_model()
