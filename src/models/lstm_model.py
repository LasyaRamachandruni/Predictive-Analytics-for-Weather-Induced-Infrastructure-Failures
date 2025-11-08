"""
lstm_model.py
--------------
LSTM model for outage prediction (temporal forecasting).
"""

import torch
import torch.nn as nn

class LSTMOutagePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMOutagePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    model = LSTMOutagePredictor(input_size=10)
    sample_input = torch.randn(5, 7, 10)  # (batch_size, time_steps, features)
    print(model(sample_input))
    print("✅ LSTM model runs successfully.")
