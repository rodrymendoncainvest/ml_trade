# =====================================================================
#  Normalizer V3 — Industrial, consistente com toda a pipeline
# =====================================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from config.paths import get_paths


# =====================================================================
# Custom Scalers — matematicamente rigorosos
# =====================================================================
class StandardScalerCustom:
    """Implementação manual de StandardScaler (mean/std)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # evitar divisões por zero
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, data: np.ndarray):
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: np.ndarray):
        return data * self.std_ + self.mean_


class RobustScalerCustom:
    """Implementação manual de RobustScaler (median / IQR)."""

    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, data: np.ndarray):
        self.median_ = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        self.iqr_ = q3 - q1
        self.iqr_[self.iqr_ == 0] = 1e-8
        return self

    def transform(self, data: np.ndarray):
        return (data - self.median_) / self.iqr_

    def inverse_transform(self, data: np.ndarray):
        return data * self.iqr_ + self.median_


# =====================================================================
# Normalizer V3 — integradíssimo com a estrutura ML_Trade
# =====================================================================
class Normalizer:
    """
    Normalizer Industrial:
    - remove velas sintéticas
    - adiciona logreturn (opcional)
    - aplica scaler robusto ou standard a OHLCV
    - salva scaler por ticker em storage/scalers/<TICKER>_scaler.pkl
    - garante dataset limpo, finito e estável
    """

    def __init__(self, scaler_type="standard", use_logreturn=True, ticker="GENERIC"):
        self.scaler_type = scaler_type
        self.use_logreturn = use_logreturn
        self.ticker = ticker.upper()

        # escolher scaler
        if scaler_type == "standard":
            self.scaler = StandardScalerCustom()
        elif scaler_type == "robust":
            self.scaler = RobustScalerCustom()
        else:
            raise ValueError("scaler_type deve ser 'standard' ou 'robust'.")

        # caminhos industriais
        paths = get_paths(self.ticker)
        self.scaler_path = Path(paths["scalers"]) / f"{self.ticker}_scaler.pkl"

    # -----------------------------------------------------------------
    # LOGRETURN seguro (sem warnings)
    # -----------------------------------------------------------------
    @staticmethod
    def compute_logreturn(series: pd.Series) -> pd.Series:
        s = series.replace(0, np.nan)
        lr = np.log(s / s.shift(1))
        lr = lr.replace([np.inf, -np.inf], np.nan).fillna(0)
        return lr

    # -----------------------------------------------------------------
    # extrair OHLCV para matriz
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_matrix(df: pd.DataFrame) -> np.ndarray:
        return df[["open", "high", "low", "close", "volume"]].values

    @staticmethod
    def _apply_scaled(df: pd.DataFrame, scaled: np.ndarray) -> pd.DataFrame:
        df = df.copy()
        df["open"], df["high"], df["low"], df["close"], df["volume"] = scaled.T
        return df

    # -----------------------------------------------------------------
    # salvar / carregar scaler
    # -----------------------------------------------------------------
    def save(self):
        joblib.dump(self.scaler, self.scaler_path)

    def load(self):
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    # -----------------------------------------------------------------
    # NORMALIZAÇÃO PRINCIPAL
    # -----------------------------------------------------------------
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1) remover velas sintéticas
        if "synthetic" in df.columns:
            synth = int(df["synthetic"].sum())
            if synth > 0:
                print(f"[Normalizer] Removidas {synth} velas sintéticas.")
                df = df[df["synthetic"] == 0].reset_index(drop=True)

        # 2) calcular logreturn
        if self.use_logreturn:
            df["logreturn"] = self.compute_logreturn(df["close"])
        else:
            df["logreturn"] = 0.0

        # 3) extrair OHLCV
        matrix = self._extract_matrix(df)

        # 4) fit + transform
        self.scaler.fit(matrix)
        scaled_matrix = self.scaler.transform(matrix)

        df = self._apply_scaled(df, scaled_matrix)

        # 5) limpeza final
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna().reset_index(drop=True)

        # 6) guardar scaler industrial
        self.save()

        return df

    # -----------------------------------------------------------------
    # reverter normalização
    # -----------------------------------------------------------------
    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(normalized)
