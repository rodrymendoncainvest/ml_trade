import numpy as np
import pandas as pd


class WindowBuilder:
    """
    WindowBuilder V3 Industrial
    - Constrói tensores temporais para modelos sequenciais.
    - Compatível com TCN / LSTM / GRU / Transformer / N-BEATS.
    - Robusto contra NaNs, shapes inconsistentes e features faltantes.

    Output:
        X → shape (N, window_size, n_features)
        y → shape (N,) ou (N, horizon)
    """

    def __init__(
        self,
        window_size: int = 64,
        horizon: int = 1,
        feature_cols: list | None = None,
        target_col: str = "close",
        enforce_strict=True
    ):
        """
        window_size: nº de velas usadas como input.
        horizon: nº de velas previstas (1 = previsão pontual).
        feature_cols: lista de features a usar.
        target_col: coluna alvo (tipicamente close ou logreturn).
        enforce_strict: força validação rigorosa do dataset.
        """
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.enforce_strict = enforce_strict

    # ============================================================
    # VALIDAÇÕES
    # ============================================================

    def _validate_columns(self, df: pd.DataFrame):
        if self.feature_cols is None:
            raise ValueError("WindowBuilder: feature_cols não especificado.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"WindowBuilder: faltam features: {missing}")

        if self.target_col not in df.columns:
            raise ValueError(f"WindowBuilder: target '{self.target_col}' não existe no DataFrame.")

    def _validate_no_nan(self, df: pd.DataFrame):
        subset = self.feature_cols + [self.target_col]
        if df[subset].isna().any().any():
            raise ValueError("WindowBuilder: NaNs encontrados nas features ou target.")

    def _validate_length(self, df: pd.DataFrame):
        required = self.window_size + self.horizon
        if len(df) < required:
            raise ValueError(
                f"WindowBuilder: dataset demasiado curto. "
                f"Existem {len(df)} linhas, mínimo necessário: {required}."
            )

    # ============================================================
    # BUILD COMPLETO
    # ============================================================

    def build(self, df: pd.DataFrame):
        df = df.reset_index(drop=True).copy()

        if self.enforce_strict:
            self._validate_columns(df)
            self._validate_no_nan(df)
            self._validate_length(df)

        features = df[self.feature_cols].values.astype(np.float32)
        target = df[self.target_col].values.astype(np.float32)

        X = []
        y = []

        max_start = len(df) - (self.window_size + self.horizon)

        for start in range(max_start):
            end = start + self.window_size
            horizon_end = end + self.horizon

            window = features[start:end]

            if self.horizon == 1:
                target_seq = target[end]
            else:
                target_seq = target[end:horizon_end]

            X.append(window)
            y.append(target_seq)

        X = np.array(X, dtype=np.float32)

        if self.horizon == 1:
            y = np.array(y, dtype=np.float32).reshape(-1)
        else:
            y = np.array(y, dtype=np.float32)

        return X, y

    # ============================================================
    # SINGLE WINDOW (real-time)
    # ============================================================

    def build_single(self, df: pd.DataFrame):
        if self.feature_cols is None:
            raise ValueError("WindowBuilder: feature_cols não definido.")

        df = df.reset_index(drop=True)

        if len(df) < self.window_size:
            raise ValueError(
                f"WindowBuilder: necessário {self.window_size} registos "
                f"mas só existem {len(df)}."
            )

        data = df[self.feature_cols].values.astype(np.float32)
        window = data[-self.window_size:]

        return np.array(window, dtype=np.float32)
