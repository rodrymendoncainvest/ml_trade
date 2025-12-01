import pandas as pd
import numpy as np


# ===================================================================
#  MAPA TICKER → MERCADO
#  Sem heurísticas. Totalmente explícito.
# ===================================================================
MARKET_OVERRIDES = {
    "GALP.LS": "EU",
    "AAPL": "US",
    "MSFT": "US",
    "BTC-USD": "CRYPTO",
    "EURUSD=X": "FOREX",
    # o utilizador pode adicionar aqui novos tickers
}


# ===================================================================
#  MAPA MERCADO → HORAS OFICIAIS DE SESSÃO
# ===================================================================
MARKET_SESSIONS = {
    "EU":  {"open": 8,   "close": 17},   # Euronext / Lisboa
    "US":  {"open": 9,   "close": 16},   # NYSE / NASDAQ (hora local ajustada pelo Yahoo)
    "CRYPTO": {"open": 0, "close": 23},  # 24/7
    "FOREX":  {"open": 0, "close": 23},  # 24/5
}


class Aligner:
    """
    Alinha timestamps para frequência uniforme, sem heurísticas exceto:
    → Remoção de fins-de-semana para equities (EU e US)

    Responsabilidades industriais:
    - resample OHLCV financeiro
    - marcação correta de velas reais vs sintéticas
    - forward-fill limitado
    - limpeza de candles impossíveis
    - compatível com Validator e Normalizer
    """

    def __init__(self, frequency="1h", fill_limit=2):
        self.frequency = frequency
        self.fill_limit = fill_limit

    # ===================================================================
    #  Determinar mercado do ticker
    # ===================================================================
    def _get_market(self, ticker: str) -> str:
        if ticker not in MARKET_OVERRIDES:
            raise ValueError(
                f"[Aligner] Ticker '{ticker}' não encontrado em MARKET_OVERRIDES. "
                f"Adiciona-o manualmente."
            )
        return MARKET_OVERRIDES[ticker]

    # ===================================================================
    #  Remover fins-de-semana apenas para equities (EU/US)
    # ===================================================================
    def _remove_weekends(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        if market in ["EU", "US"]:
            df = df[df["timestamp"].dt.weekday <= 4]  # 0–4 = segunda–sexta
        return df

    # ===================================================================
    #  Filtrar horas de sessão (não é heurística: horário fixo de mercado)
    # ===================================================================
    def _filter_session(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        sess = MARKET_SESSIONS[market]

        df["hour"] = df["timestamp"].dt.hour
        df = df[(df["hour"] >= sess["open"]) & (df["hour"] <= sess["close"])]
        df = df.drop(columns=["hour"])

        return df

    # ===================================================================
    #  Resample financeiro OHLCV
    # ===================================================================
    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("timestamp")

        out = df.resample(self.frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        out["synthetic"] = 1  # tudo começa marcado como artificial

        return out

    # ===================================================================
    #  Identificar candles reais
    # ===================================================================
    def _mark_real(self, df_resampled, df_original):
        real_index = set(df_original["timestamp"].values)
        rs_index = df_resampled.index.values

        df_resampled["synthetic"] = np.array(
            [0 if ts in real_index else 1 for ts in rs_index]
        )

        return df_resampled

    # ===================================================================
    #  Forward-fill limitado
    # ===================================================================
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(
            limit=self.fill_limit
        )

        df["volume"] = df["volume"].fillna(0)  # volume válido para sintéticos
        df["synthetic"] = df["synthetic"].fillna(1).astype(int)

        return df

    # ===================================================================
    #  Limpeza final
    # ===================================================================
    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # velas impossíveis
        df = df[df["high"] >= df["low"]]

        # OHLC críticos
        df = df.dropna(subset=["open", "high", "low", "close"])

        df = df.sort_index()
        df = df.reset_index()  # timestamp volta a coluna

        return df

    # ===================================================================
    #  Função principal
    # ===================================================================
    def align(self, df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Aligner: df deve ser DataFrame.")

        if "timestamp" not in df.columns:
            raise ValueError("Aligner: falta coluna 'timestamp'.")

        if ticker is None:
            raise ValueError("Aligner: é necessário fornecer o ticker para determinar o mercado.")

        # 1) mercado
        market = self._get_market(ticker)

        df_local = df.copy()

        # 2) cortar fins-de-semana (equities)
        df_local = self._remove_weekends(df_local, market)

        # 3) cortar horas fora da sessão
        df_local = self._filter_session(df_local, market)

        # 4) resample financeiro
        df_resampled = self._resample(df_local)

        # 5) marcar velas reais
        df_resampled = self._mark_real(df_resampled, df_local)

        # 6) forward-fill controlado
        df_filled = self._fill_gaps(df_resampled)

        # 7) limpeza final
        df_final = self._cleanup(df_filled)

        return df_final


# Instância global
aligner = Aligner()
