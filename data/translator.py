import pandas as pd
import numpy as np


class Translator:
    """
    Traduz dados RAW provenientes de diferentes providers
    e garante um formato interno padronizado:

    - timestamp (UTC)
    - open, high, low, close, volume
    - ordenação temporal
    - remoção de velas impossíveis
    - limpeza de duplicados

    Compatível com:
    - RawIngestor (CSV ingestion)
    - DataDownloader (Yahoo 1H)
    - Validator industrial
    - Normalizer (logreturns + scalers)
    - Feature generator (10k linhas)
    """

    REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]

    COLUMN_MAP = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",

        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",

        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    # ---------------------------------------------------------
    # 1. Renomear colunas
    # ---------------------------------------------------------
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self.COLUMN_MAP.get(col, col).lower() for col in df.columns]
        return df

    # ---------------------------------------------------------
    # 2. Timestamp → UTC
    # ---------------------------------------------------------
    def _normalize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError("Translator: falta coluna 'timestamp'.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if df["timestamp"].isna().any():
            raise ValueError("Translator: timestamps inválidos no RAW.")

        # garantir timezone UTC (mesmo se já tiver)
        try:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        except Exception:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        return df

    # ---------------------------------------------------------
    # 3. Forçar OHLCV
    # ---------------------------------------------------------
    def _force_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.REQUIRED:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.REQUIRED]

    # ---------------------------------------------------------
    # 4. Limpeza inicial
    # ---------------------------------------------------------
    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # vela impossível
        df = df[df["high"] >= df["low"]]

        # remover NaNs estruturais
        df = df.dropna(subset=["open", "high", "low", "close"])

        return df

    # ---------------------------------------------------------
    # Função principal
    # ---------------------------------------------------------
    def translate(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("Translator: df_raw deve ser DataFrame.")

        df = df_raw.copy()
        df = self._rename_columns(df)
        df = self._normalize_timestamp(df)
        df = self._force_ohlcv(df)
        df = self._cleanup(df)

        return df


translator = Translator()
