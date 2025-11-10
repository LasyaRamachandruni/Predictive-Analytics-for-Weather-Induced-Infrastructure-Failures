import torch

from src.models.lstm_model import LSTMOutagePredictor


def test_lstm_forward_shape():
    model = LSTMOutagePredictor(input_size=6, hidden_size=32, num_layers=1, dropout=0.0)
    dummy_input = torch.randn(4, 12, 6)
    output = model(dummy_input)
    assert output.shape == (4, 1)
