# lstm.py
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM Industrial — forecasting multi-step.

    - Input:  (batch, window, features)
    - Output: (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ============================================================
        # STACK LSTM
        # ============================================================
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ============================================================
        # CAMADA FINAL — regressão multi-step
        # ============================================================
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        """
        x: (batch, window, features)
        """

        # LSTM output: (batch, window, hidden_dim)
        out, _ = self.lstm(x)

        # pegar a última posição temporal
        last = out[:, -1, :]  # (batch, hidden_dim)

        # regressão multi-step
        pred = self.fc(last)  # (batch, horizon)

        return pred

    @torch.no_grad()
    def predict(self, x):
        """
        Predição segura.
        """
        self.eval()
        return self.forward(x)
