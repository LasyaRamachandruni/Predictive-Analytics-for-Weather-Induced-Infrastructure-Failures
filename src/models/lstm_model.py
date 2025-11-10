"""
PyTorch LSTM model for modelling temporal failure risk.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMOutagePredictor(nn.Module):
    """
    A configurable LSTM backbone followed by a linear projection.

    Parameters
    ----------
    input_size:
        Number of features per timestep.
    hidden_size:
        Hidden dimension of the LSTM layers.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers.
    output_size:
        Size of the output projection. Use 1 for regression or binary classification.
    bidirectional:
        Whether to use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x:
            Tensor of shape `(batch_size, seq_len, input_size)`.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


def count_parameters(model: nn.Module) -> int:
    """Utility to count trainable parameters (useful for logging)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
