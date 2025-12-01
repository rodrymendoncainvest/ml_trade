import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Converte previsões ML em sinais {-1,0,1}.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def from_predictions(self, predictions: np.ndarray) -> pd.Series:
        """
        regressão:
          > threshold → long
          < -threshold → short
          caso contrário → flat
        """
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        signals = np.zeros(len(predictions), dtype=int)

        signals[predictions > self.threshold] = 1
        signals[predictions < -self.threshold] = -1

        return pd.Series(signals)
