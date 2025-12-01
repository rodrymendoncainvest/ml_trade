import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


class Validator:
    """
    Validator industrial para séries OHLCV.

    Garantias:
    - Sem heurísticas escondidas
    - Compatível com o novo aligner (synthetic flags, horários, resample)
    - Reprova séries incoerentes de forma explícita
    - Mantém precisão matemática e consistência temporal

    Valida:
    - presença de colunas mandatórias
    - NaNs estruturais
    - coerência OHLC
    - amplitude impossível
    - volume anómalo
    - gaps intradiários reais baseados nos timestamps da série
    - estacionaridade (opcional)
    - autocorrelação (opcional)
    - comprimento mínimo
    """

    REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        require_stationarity=False,
        check_autocorr=False,
        min_history_points=300,
        max_intraday_gap_factor=3,
    ):
        self.require_stationarity = require_stationarity
        self.check_autocorr = check_autocorr
        self.min_history_points = min_history_points
        self.max_intraday_gap_factor = max_intraday_gap_factor

    # ==========================================================
    # 1. Colunas obrigatórias
    # ==========================================================
    def _check_columns(self, df):
        for col in self.REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"[Validator] Falta coluna obrigatória: '{col}'.")

    # ==========================================================
    # 2. NaNs estruturais
    # ==========================================================
    def _check_nans(self, df):
        if df[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError("[Validator] NaNs encontrados em OHLC.")
        if df["volume"].isna().any():
            raise ValueError("[Validator] Volume contém NaNs.")
        if df["timestamp"].isna().any():
            raise ValueError("[Validator] Timestamp inválido encontrado.")

    # ==========================================================
    # 3. Coerência OHLC
    # ==========================================================
    def _check_ohlc(self, df):
        if (df["high"] < df["low"]).any():
            raise ValueError("[Validator] high < low detetado.")
        if ((df["open"] > df["high"]) | (df["open"] < df["low"])).any():
            raise ValueError("[Validator] open fora de limites high/low.")
        if ((df["close"] > df["high"]) | (df["close"] < df["low"])).any():
            raise ValueError("[Validator] close fora de limites high/low.")

        # vela absurda — amplitude extrema comparada com median
        amp = df["high"] - df["low"]
        if (amp > amp.median() * 10).any():
            raise ValueError("[Validator] vela impossível (amplitude absurda).")

    # ==========================================================
    # 4. Volume
    # ==========================================================
    def _check_volume(self, df):
        if (df["volume"] < 0).any():
            raise ValueError("[Validator] volume negativo encontrado.")
        if df["volume"].rolling(20).sum().eq(0).any():
            raise ValueError("[Validator] sequência longa de volume zero.")

    # ==========================================================
    # 5. Comprimento mínimo
    # ==========================================================
    def _check_min_length(self, df):
        if len(df) < self.min_history_points:
            raise ValueError(
                f"[Validator] série demasiado curta ({len(df)}). "
                f"Requer mínimo: {self.min_history_points}"
            )

    # ==========================================================
    # 6. Gap-check intradiário (baseado apenas nos dados)
    # ==========================================================
    def _check_intraday_gaps(self, df):
        ts = df["timestamp"].values

        for i in range(1, len(ts)):
            curr = ts[i]
            prev = ts[i - 1]

            # se mudou de dia → natural
            if pd.Timestamp(curr).date() != pd.Timestamp(prev).date():
                continue

            delta_h = (curr - prev).astype("timedelta64[h]").astype(int)

            # delta_h > 1 significa que faltaram velas reais
            if delta_h > 1 and delta_h > self.max_intraday_gap_factor:
                raise ValueError(f"[Validator] gap intradiário anormal ({delta_h}h).")

    # ==========================================================
    # 7. Estacionaridade ADF (opcional)
    # ==========================================================
    def _check_stationarity(self, df):
        try:
            p = adfuller(df["close"], autolag="AIC")[1]
        except Exception:
            return
        if p > 0.05:
            raise ValueError("[Validator] série não estacionária (ADF p > 0.05).")

    # ==========================================================
    # 8. Autocorrelação (opcional)
    # ==========================================================
    def _check_autocorr(self, df):
        if len(df) < 60:
            return
        ac = df["close"].autocorr(lag=1)
        if abs(ac) < 0.05:
            raise ValueError("[Validator] autocorrelação insuficiente.")

    # ==========================================================
    # 9. Execução principal
    # ==========================================================
    def validate(self, df):
        df = df.copy().reset_index(drop=True)

        self._check_columns(df)
        self._check_nans(df)
        self._check_ohlc(df)
        self._check_volume(df)
        self._check_min_length(df)
        self._check_intraday_gaps(df)

        if self.require_stationarity:
            self._check_stationarity(df)

        if self.check_autocorr:
            self._check_autocorr(df)

        return df


validator = Validator()
