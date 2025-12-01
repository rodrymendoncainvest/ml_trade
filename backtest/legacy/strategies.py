import pandas as pd
import numpy as np


class MLStrategy:
    """
    Estratégia simples baseada em previsões ML.

    Regras:
    - A previsão target > 0 → BUY
    - target < 0 → SELL
    - zona morta → FLAT
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def generate_signals(self, predictions: np.ndarray) -> pd.Series:
        signals = np.zeros(len(predictions), dtype=int)
        signals[predictions > self.threshold] = 1
        signals[predictions < -self.threshold] = -1
        return pd.Series(signals, name="signal")
