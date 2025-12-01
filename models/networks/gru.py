# gru.py
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU Industrial para forecasting financeiro.

    Input:
        x → (batch, window, features)

    Output:
        y → (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU batch-first (fundamental)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # FC final para previsão multi-step
        self.fc = nn.Linear(hidden_dim, horizon)

        # Inicialização industrial
        self._init_weights()

    # ------------------------------------------------------------
    # Inicialização correta (ganho para RNNs)
    # ------------------------------------------------------------
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    def forward(self, x):
        """
        x: (batch, window, features)
        """
        # h0 inicializado a zeros (não reutilizamos estado entre batches)
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)

        out, _ = self.gru(x, h0)

        # última posição temporal → forecasting
        last = out[:, -1, :]  # (batch, hidden_dim)

        return self.fc(last)  # (batch, horizon)

    # ------------------------------------------------------------
    # PREVISÃO INDUSTRIAL
    # ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.forward(x)
