import numpy as np
import pandas as pd


class WindowBuilder:
    """
    WindowBuilder industrial para modelos sequenciais (TCN, LSTM,
    GRU, Transformer, N-BEATS).

    - Cria janelas deslizantes X com shape (N, window, features)
    - Cria targets y com multi-step (horizon)
    - Compatível com build_single para inferência
    """

    def __init__(self, window_size, horizon, feature_cols, target_col="close"):
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.feature_cols = feature_cols
        self.target_col = target_col

        if not isinstance(self.feature_cols, list):
            raise ValueError("WindowBuilder: feature_cols deve ser lista de strings.")

        if len(self.feature_cols) == 0:
            raise ValueError("WindowBuilder: feature_cols vazio.")

    # -----------------------------------------------------------
    #  VALIDATE DATAFRAME LENGTH
    # -----------------------------------------------------------
    def _validate_length(self, df: pd.DataFrame):
        required = self.window_size + self.horizon
        if len(df) < required:
            raise ValueError(
                f"WindowBuilder: dataset demasiado curto. "
                f"Existem {len(df)} linhas, mínimo necessário: {required}."
            )

    # -----------------------------------------------------------
    #  BUILD FULL DATASET (TRAINING)
    # -----------------------------------------------------------
    def build(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)

        self._validate_length(df)

        data = df[self.feature_cols].values.astype(np.float32)
        target = df[self.target_col].values.astype(np.float32)

        X_list = []
        y_list = []

        limit = len(df) - self.window_size - self.horizon + 1

        for i in range(limit):
            window = data[i : i + self.window_size]
            future = target[i + self.window_size : i + self.window_size + self.horizon]

            X_list.append(window)
            y_list.append(future)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y

    # -----------------------------------------------------------
    #  BUILD SINGLE WINDOW (INFERENCE)
    # -----------------------------------------------------------
    def build_single(self, df: pd.DataFrame) -> np.ndarray:
        """
        Usado apenas na inferência.

        Retorna última janela com shape (window_size, features)
        """

        df = df.reset_index(drop=True)

        if len(df) < self.window_size:
            raise ValueError(
                f"WindowBuilder: necessário {self.window_size} registos "
                f"mas só existem {len(df)}."
            )

        data = df[self.feature_cols].values.astype(np.float32)

        window = data[-self.window_size :]

        return np.array(window, dtype=np.float32)
