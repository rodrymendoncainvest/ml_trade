import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class FeatureScaler:
    """
    Escalador industrial para todas as features pós-FeatureGenerator.
    - Fit apenas no preprocess
    - Transform apenas na inferência/backtest
    - Mantém mean/std persistidos em disco
    - 100% determinístico
    """

    def __init__(self, scaler_path: Path):
        self.scaler_path = scaler_path
        self.mean_ = None
        self.std_ = None

    # -------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """Faz fit e guarda mean/std em disco."""
        data = df.values.astype(np.float32)

        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1e-8

        joblib.dump(
            {"mean": self.mean_, "std": self.std_, "columns": list(df.columns)},
            self.scaler_path
        )

    # -------------------------------------------------------------
    # TRANSFORM
    # -------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carrega mean/std do disco e aplica transform.
        Garante que a ordem das colunas é a mesma.
        """
        if self.mean_ is None:
            if not self.scaler_path.exists():
                raise FileNotFoundError(
                    f"[FeatureScaler] scaler não encontrado: {self.scaler_path}\n"
                    "Corre o preprocess primeiro."
                )

            obj = joblib.load(self.scaler_path)
            self.mean_ = obj["mean"]
            self.std_ = obj["std"]
            saved_cols = obj.get("columns", None)

            if saved_cols is not None:
                # reordenar para garantir coerência treino/inferência
                df = df[saved_cols]

        data = df.values.astype(np.float32)
        scaled = (data - self.mean_) / self.std_

        return pd.DataFrame(scaled, columns=df.columns)

    # -------------------------------------------------------------
    # FIT + TRANSFORM (PREPROCESS)
    # -------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usado apenas no preprocess."""
        self.fit(df)
        return self.transform(df)
