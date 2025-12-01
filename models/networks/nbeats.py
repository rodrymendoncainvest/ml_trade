# nbeats.py
import torch
import torch.nn as nn


# ============================================================
# BLOCO BÁSICO N-BEATS
# ============================================================
class NBeatsBlock(nn.Module):
    """
    Bloco fundamental do N-BEATS:
        - MLP profundo
        - Backcast + Forecast heads
        - Residual learning
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        theta_dim: int,
        nb_layers: int = 4,
        layer_size: int = 512,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(nb_layers):
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.ReLU())
            in_dim = layer_size

        self.fc = nn.Sequential(*layers)

        # parâmetros do bloco
        self.backcast_fc = nn.Linear(layer_size, input_dim)
        self.forecast_fc = nn.Linear(layer_size, theta_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_in = x
        x = x.view(x.size(0), -1)  # flatten → (batch, window*features)

        h = self.fc(x)

        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)

        return backcast, forecast


# ============================================================
# MODELO N-BEATS COMPLETO (GENERIC STACK)
# ============================================================
class NBeatsModel(nn.Module):
    """
    N-BEATS Industrial para forecasting financeiro.

    Input:
        x → (batch, window, features)

    Output:
        y → (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,      # features
        window_size: int,    # histórico
        horizon: int = 1,
        hidden_dim: int = 512,
        num_blocks: int = 4,     # mais blocos = mais potência
        nb_layers: int = 4,
        layer_size: int = 512,
    ):
        super().__init__()

        self.window_size = window_size
        self.input_dim = input_dim
        self.horizon = horizon

        # dimensão de entrada para cada bloco
        block_input_dim = window_size * input_dim
        theta_dim = horizon

        # stacks
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=block_input_dim,
                hidden_dim=hidden_dim,
                theta_dim=theta_dim,
                nb_layers=nb_layers,
                layer_size=layer_size,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        x: (batch, window, features)
        """
        x = x.reshape(x.size(0), -1)   # flatten

        residual = x
        forecast_final = 0

        for block in self.blocks:
            backcast, forecast = block(residual)

            residual = residual - backcast
            forecast_final = forecast_final + forecast

        return forecast_final  # (batch, horizon)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.forward(x)
