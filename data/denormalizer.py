# data/denormalizer.py

import numpy as np


class PriceDenormalizer:
    """
    Inversão correta para previsões ML:
    - Só usa mean/std da coluna 'close'
    - Evita explosões ao aplicar inversão OHLCV a forecast escalar
    """

    def __init__(self, scaler):
        # scaler.mean_ → [open, high, low, close, volume]
        # scaler.std_  → [open, high, low, close, volume]
        self.mean_close = scaler.mean_[3]
        self.std_close = scaler.std_[3]

    def inverse(self, pred_value):
        """
        Recebe valor predito (normalizado) → devolve preço real.
        """
        # pred_value pode vir como numpy, float32, tensor, etc.
        val = float(pred_value)
        return val * self.std_close + self.mean_close
