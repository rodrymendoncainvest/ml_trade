# =====================================================================
#  Normalizer V5 — Coerente, funcional e 100% estável
# =====================================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from config.paths import get_paths


# =====================================================================
#  SCALERS
# =====================================================================

class StandardScalerCustom:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, data):
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        return data * self.std_ + self.mean_


class RobustScalerCustom:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, data):
        self.median_ = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1e-8
        self.iqr_ = iqr
        return self

    def transform(self, data):
        return (data - self.median_) / self.iqr_

    def inverse_transform(self, data):
        return data * self.iqr_ + self.median_


# =====================================================================
#  NORMALIZER
# =====================================================================

class Normalizer:
    """
    Corrigido:
    - Usa SEMPRE object save/load (nunca dict)
    - Nunca mistura formatos
    - Evita erro 'object not subscriptable'
    """

    def __init__(self, scaler_type="standard", use_logreturn=True, ticker="GENERIC"):
        self.scaler_type = scaler_type
        self.use_logreturn = use_logreturn
        self.ticker = ticker.upper()

        if scaler_type == "standard":
            self.scaler = StandardScalerCustom()
        else:
            self.scaler = RobustScalerCustom()

        paths = get_paths(self.ticker)
        self.scaler_path = Path(paths["scalers"]) / f"{self.ticker}_scaler.pkl"

    # ---------------------------
    # INTERNAL: save / load
    # ---------------------------

    def save_scaler(self):
        joblib.dump(self.scaler, self.scaler_path)

    def load_scaler(self) -> bool:
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    # ---------------------------

    @staticmethod
    def compute_logreturn(series):
        s = series.replace(0, np.nan)
        lr = np.log(s / s.shift(1))
        return lr.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def _extract_matrix(df):
        return df[["open", "high", "low", "close", "volume"]].values

    @staticmethod
    def _apply_scaled(df, scaled):
        df2 = df.copy()
        df2["open"], df2["high"], df2["low"], df2["close"], df2["volume"] = scaled.T
        return df2

    # =================================================================
    # MAIN NORMALIZATION
    # =================================================================

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Remove velas sintéticas
        if "synthetic" in df.columns:
            df = df[df["synthetic"] == 0].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Normalizer: dataset vazio após remover sintéticas.")

        # logreturn
        df["logreturn"] = (
            self.compute_logreturn(df["close"])
            if self.use_logreturn else 0.0
        )

        matrix = self._extract_matrix(df)

        # Fit sempre em preprocess — nunca tenta load aqui
        self.scaler.fit(matrix)
        scaled = self.scaler.transform(matrix)

        df_out = self._apply_scaled(df, scaled)

        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_out.dropna(inplace=True)
        df_out.reset_index(drop=True, inplace=True)

        self.save_scaler()

        return df_out

    # =================================================================
    # INVERSE
    # =================================================================

    def inverse_close(self, norm_value):
        """
        Inversão só da coluna close.
        Usada pela PriceDenormalizer.
        """
        idx = 3  # posição da coluna close
        return float(norm_value * self.scaler.std_[idx] + self.scaler.mean_[idx])
