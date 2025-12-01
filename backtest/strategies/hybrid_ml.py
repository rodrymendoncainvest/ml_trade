import numpy as np
import pandas as pd


class HybridMLStrategy:
    """
    HybridMLStrategy V1 — Estratégia Industrial Híbrida

    Combina:
    ----------------------------------
    1) Sinal do modelo ML (coluna 'prediction')
    2) Sinal técnico simples baseado em tendência
    3) Fusão determinística e transparente → sinal final

    Regras:
    ----------------------------------
    - modelo acima do threshold → pressão LONG
    - modelo abaixo do -threshold → pressão SHORT
    - sinal técnico confirma a direção
    - se divergirem → reduz-se posição (FLAT)
    """

    def __init__(self, threshold=0.001):
        self.threshold = threshold

    # ================================================================
    # 1) Sinal técnico — tendência mínima (EMA20 - EMA50)
    # ================================================================
    def _technical_signal(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("HybridMLStrategy: falta coluna 'close'.")

        # EMAs
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()

        trend = ema20 - ema50

        tech_sig = np.where(
            trend > 0, 1,
            np.where(trend < 0, -1, 0)
        )

        return pd.Series(tech_sig, index=df.index, name="tech_signal")

    # ================================================================
    # 2) Sinal do modelo ML
    # ================================================================
    def _prediction_signal(self, df: pd.DataFrame) -> pd.Series:
        if "prediction" not in df.columns:
            raise ValueError("HybridMLStrategy: falta coluna 'prediction'.")

        pred = df["prediction"]

        pred_sig = np.where(
            pred > self.threshold, 1,
            np.where(pred < -self.threshold, -1, 0)
        )

        return pd.Series(pred_sig, index=df.index, name="ml_signal")

    # ================================================================
    # 3) Fusão dos sinais
    # ================================================================
    def _fuse(self, ml_sig: pd.Series, tech_sig: pd.Series) -> pd.Series:
        ml = ml_sig.values
        tc = tech_sig.values

        fused = []

        for m, t in zip(ml, tc):

            # ambos concordam → posição plena
            if m == t:
                fused.append(m)

            # um está neutro → segue o outro
            elif m == 0:
                fused.append(t)
            elif t == 0:
                fused.append(m)

            # divergência → neutralidade
            else:
                fused.append(0)

        return pd.Series(fused, index=ml_sig.index, name="signal")

    # ================================================================
    # 4) API pública compatível com BacktestEngine V4
    # ================================================================
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Recebe: DataFrame completo (OHLCV + features + prediction)
        Retorna: Série de sinais (-1, 0, 1)
        """

        tech = self._technical_signal(df)
        ml = self._prediction_signal(df)

        return self._fuse(ml, tech)
