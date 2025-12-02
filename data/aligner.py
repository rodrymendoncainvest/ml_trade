import pandas as pd
import numpy as np

"""
aligner.py — Alinhador Industrial OHLCV

Objetivo:
- Trabalhar sobre dados já em 1H (Yahoo Finance)
- Tapar buracos intradiários com velas sintéticas
- Manter velas reais intactas
- Respeitar horários de mercado em UTC
- Fornecer coluna 'synthetic' para o Normalizer

Fluxo:
    RAW (1H Yahoo) → filtra sessão (UTC) → resample 1H + gaps → marca synthetic
"""


# ===================================================================
#  MAPA TICKER → MERCADO (Prioridade máxima)
# ===================================================================
MARKET_OVERRIDES = {
    "GALP.LS": "EU",
    "AAPL": "US",
    "MSFT": "US",
    "BTC-USD": "CRYPTO",
    "EURUSD=X": "FOREX",
    # podes adicionar aqui overrides manuais
}


# ===================================================================
#  HORÁRIOS DE MERCADO EM UTC
#  Yahoo devolve timestamps em UTC, por isso filtramos diretamente em UTC.
# ===================================================================
MARKET_SESSIONS_UTC = {
    "EU":     {"open": 8,   "close": 17},  # aprox. 9–18 local
    "US":     {"open": 14,  "close": 21},  # 9:30–16:00 NY ≈ 14–21 UTC
    "CRYPTO": {"open": 0,   "close": 23},  # 24/7
    "FOREX":  {"open": 0,   "close": 23},  # 24/5 (tratado como 24/7 aqui)
}


class Aligner:
    """
    Alinha timestamps e constrói séries OHLCV limpas e uniformes.

    Mantém a filosofia original:
    - resample 1H
    - preenche buracos com velas sintéticas
    - 'synthetic' = 0 para velas reais, 1 para velas inventadas
    """

    def __init__(self, frequency: str = "1h", fill_limit: int = 2):
        self.frequency = frequency
        self.fill_limit = fill_limit

    # ===================================================================
    #  Auto-detecção de mercado (fallback para tickers não mapeados)
    # ===================================================================
    def _auto_detect_market(self, ticker: str) -> str:
        t = ticker.upper()

        # CRYPTO — padrão tipo BTC-USD, ETH-USD, etc.
        if "-USD" in t:
            return "CRYPTO"

        # FOREX — pares estilo EURUSD=X
        if "=" in t:
            return "FOREX"

        # US equities — tickers sem sufixo (AAPL, NVDA, ADBE, etc.)
        if "." not in t:
            return "US"

        # Euronext: AS, PA, LS, BR, MI
        if t.endswith(".AS") or t.endswith(".PA") or t.endswith(".LS") \
           or t.endswith(".BR") or t.endswith(".MI"):
            return "EU"

        # Fallback seguro
        return "EU"

    def _get_market(self, ticker: str) -> str:
        if ticker in MARKET_OVERRIDES:
            market = MARKET_OVERRIDES[ticker]
        else:
            market = self._auto_detect_market(ticker)
            print(f"[Aligner] Mercado detectado automaticamente para {ticker}: {market}")
        return market

    # ===================================================================
    #  Remover fins-de-semana (só EU/US)
    # ===================================================================
    def _remove_weekends(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        if market in ["EU", "US"]:
            df = df[df["timestamp"].dt.weekday <= 4]
        return df

    # ===================================================================
    #  Filtrar horas dentro da sessão (em UTC)
    # ===================================================================
    def _filter_session(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        session = MARKET_SESSIONS_UTC[market]
        df = df.copy()

        df["hour_utc"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

        df = df[
            (df["hour_utc"] >= session["open"]) &
            (df["hour_utc"] <= session["close"])
        ].copy()

        df.drop(columns=["hour_utc"], inplace=True)
        return df

    # ===================================================================
    #  Resample financeiro OHLCV + synthetic via agregação
    # ===================================================================
    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Truque-chave:
        - marcamos as velas originais com synthetic=0
        - resample 1H com 'synthetic' = min()
            → se existir pelo menos 1 vela real no bin → 0
            → se o bin estiver vazio → NaN → depois passa a 1 (sintética)
        """

        df = df.copy()
        df = df.set_index("timestamp")

        out = df.resample(self.frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "synthetic": "min",
        })

        return out

    # ===================================================================
    #  Forward fill limitado + marcação final de sintéticas
    # ===================================================================
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # velas que não tinham dado nenhum no bin → synthetic NaN
        # vamos marcar isso como 1 (sintética) ANTES de fill dos preços
        df["synthetic"] = df["synthetic"].fillna(1)

        # agora ffill OHLC com limite
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].ffill(limit=self.fill_limit)

        # volume vazio é 0
        df["volume"] = df["volume"].fillna(0)

        # garantir inteiro
        df["synthetic"] = df["synthetic"].astype(int)

        return df

    # ===================================================================
    #  Limpeza final
    # ===================================================================
    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        # remover candles impossíveis
        df = df[df["high"] >= df["low"]]

        # OHLC obrigatórios
        df = df.dropna(subset=["open", "high", "low", "close"])

        df = df.sort_index()
        df = df.reset_index()  # traz 'timestamp' de volta a coluna

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
            raise ValueError("Aligner: é necessário fornecer o ticker.")

        # 1) garantir timestamp em UTC
        df_local = df.copy()
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)

        # 2) mercado
        market = self._get_market(ticker)

        # 3) remover fins-de-semana
        df_local = self._remove_weekends(df_local, market)

        # 4) filtrar sessão em UTC
        df_local = self._filter_session(df_local, market)

        if df_local.empty:
            raise ValueError(
                f"[Aligner] Após filtro de sessão ({market}) não ficou nenhum candle. "
                f"Isto indica problema de timezone ou dados."
            )

        # 5) marcar todas as velas originais como reais
        df_local["synthetic"] = 0

        # 6) resample 1H com agregação
        df_resampled = self._resample(df_local)

        # 7) preencher gaps de forma controlada
        df_filled = self._fill_gaps(df_resampled)

        # 8) limpeza final
        df_final = self._cleanup(df_filled)

        return df_final


# Instância global opcional
aligner = Aligner()
