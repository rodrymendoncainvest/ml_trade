import torch
import torch.nn as nn
import math


# ============================================================
# Helpers
# ============================================================

def init_linear(layer):
    """Initialização recomendada no N-BEATS."""
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# ============================================================
# BLOCO BASE
# ============================================================

class NBeatsBlock(nn.Module):
    """
    Bloco N-BEATS refinado:
    - MLP profundo
    - Weight Normalization
    - Dropout financeiro
    - Backcast + Forecast heads
    """

    def __init__(
        self,
        input_dim: int,
        theta_dim: int,
        n_layers: int = 4,
        layer_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            inp = input_dim if i == 0 else layer_size
            dense = nn.Linear(inp, layer_size)
            init_linear(dense)

            layers.append(nn.utils.weight_norm(dense))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.fc = nn.Sequential(*layers)

        # heads
        self.backcast_fc = nn.Linear(layer_size, input_dim)
        self.forecast_fc = nn.Linear(layer_size, theta_dim)

        init_linear(self.backcast_fc)
        init_linear(self.forecast_fc)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        return backcast, forecast


# ============================================================
# FULL N-BEATS MODEL — V4
# ============================================================

class NBeatsModel(nn.Module):
    """
    N-BEATS V4 Finance-ready:
    - Generic + Trend stacks
    - MLP profundo
    - Forecast robusto em horizontes pequenos (1-step)
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        horizon: int = 1,
        num_blocks_generic: int = 3,
        num_blocks_trend: int = 3,
        layer_size: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.horizon = horizon

        block_dim = input_dim * window_size
        theta_dim = horizon

        # ---------------------------------------------------------
        # GENÉRIC STACK (capta padrões arbitrários)
        # ---------------------------------------------------------
        self.generic_blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=block_dim,
                theta_dim=theta_dim,
                n_layers=n_layers,
                layer_size=layer_size,
                dropout=dropout,
            )
            for _ in range(num_blocks_generic)
        ])

        # ---------------------------------------------------------
        # TREND STACK (capta drift + tendência lenta)
        # ---------------------------------------------------------
        self.trend_blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=block_dim,
                theta_dim=theta_dim,
                n_layers=n_layers,
                layer_size=layer_size,
                dropout=dropout,
            )
            for _ in range(num_blocks_trend)
        ])

    # ============================================================
    # FORWARD
    # ============================================================
    def forward(self, x):
        """
        x: (batch, window, features)
        """
        batch = x.size(0)
        x_flat = x.reshape(batch, -1)   # (batch, window*features)

        residual = x_flat
        forecast_final = 0

        # --- GENERIC STACK ---
        for block in self.generic_blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_final = forecast_final + forecast

        # --- TREND STACK ---
        for block in self.trend_blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_final = forecast_final + forecast

        return forecast_final  # (batch, horizon)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.forward(x)
