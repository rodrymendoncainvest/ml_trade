# transformer.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding clássico de Vaswani et al,
    adaptado para séries temporais contínuas.
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerModel(nn.Module):
    """
    Transformer Encoder Industrial para forecasting.

    Input:
        x → (batch, window, features)
    Output:
        pred → (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.horizon = horizon

        # Linear embedding → projeta features para dimensão d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Regressor final → previsão multi-step
        self.regressor = nn.Linear(d_model, horizon)

    def forward(self, x):
        """
        x: (batch, window, features)
        """

        # Embedding
        x = self.input_projection(x)  # (batch, window, d_model)

        # Add positional encodings
        x = self.pos_encoder(x)

        # Transformer encoder
        encoded = self.encoder(x)  # (batch, window, d_model)

        # Usamos a última posição temporal
        last = encoded[:, -1, :]  # (batch, d_model)

        pred = self.regressor(last)  # (batch, horizon)

        return pred

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.forward(x)
