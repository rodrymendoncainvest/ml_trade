# tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# TCN Residual Block — versão industrial
# ======================================================================
class TCNResidualBlock(nn.Module):
    """
    Bloco residual TCN com:
        - causal convolution
        - dilated kernels
        - Dropout
        - opção de projection-shortcut quando os canais mudam
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual projection se o número de canais mudar
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.padding = padding

    def _remove_future_padding(self, x, pad):
        """
        Em convoluções causais, o padding duplica informação futura.
        Aqui retiramos o padding da direita para remover o leakage.
        """
        if pad == 0:
            return x
        return x[:, :, :-pad]

    def forward(self, x):
        out = self.conv1(x)
        out = self._remove_future_padding(out, self.padding)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self._remove_future_padding(out, self.padding)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ======================================================================
# MODELO TCN COMPLETO (V4)
# ======================================================================
class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    Industrial-grade para forecasting financeiro.

    Input:
        x → (batch, window, features)

    Output:
        (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        if kernel_size < 2:
            raise ValueError("kernel_size deve ser >= 2.")

        self.horizon = horizon

        layers = []
        in_channels = input_dim

        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx

            layers.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

            in_channels = hidden_dim

        self.network = nn.Sequential(*layers)

        # proibição total de regressão autoregressiva → single shot forecasting
        self.output_layer = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        """
        x: (batch, window, features)
        """
        # converter para (batch, channels, window)
        x = x.transpose(1, 2)

        out = self.network(x)

        # última posição temporal
        last = out[:, :, -1]

        # previsão multi-step
        return self.output_layer(last)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.forward(x)
