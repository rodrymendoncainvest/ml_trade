import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


class Validator:
    """
    Validator industrial robusto para OHLCV após resample 1H.

    Ajustado para:
    - gaps reais grandes (EU/US)
    - resample com candles sintéticos
    - mercados fechados
    - volatilidade irregular
    - dados Yahoo Finance
    - candles outliers mas legítimos

    Reprova apenas:
    - incoerências estruturais (open>high, close<low, etc.)
    - amplitude fisicamente impossível (open/close fora do range)
    - volume negativo
    - NaNs estruturais
    - gaps absurdos intraday
    - comprimento insuficiente
    """

    REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        require_stationarity=False,
        check_autocorr=False,
        min_history_points=300,
        max_intraday_gap_hours=12,
        max_amp_factor=200,  # agressivamente permissivo
    ):
        self.require_stationarity = require_stationarity
        self.check_autocorr = check_autocorr
        self.min_history_points = min_history_points
        self.max_intraday_gap_hours = max_intraday_gap_hours
        self.max_amp_factor = max_amp_factor

    # ==========================================================
    # 1. Colunas obrigatórias
    # ==========================================================
    def _check_columns(self, df):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[Validator] Faltam colunas obrigatórias: {missing}")

    # ==========================================================
    # 2. NaNs estruturais
    # ==========================================================
    def _check_nans(self, df):
        if df[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError("[Validator] OHLC contém NaNs.")
        if df["volume"].isna().any():
            raise ValueError("[Validator] Volume contém NaNs.")
        if df["timestamp"].isna().any():
            raise ValueError("[Validator] Timestamp inválido.")

    # ==========================================================
    # 3. Coerência OHLC fundamental
    # ==========================================================
    def _check_ohlc_consistency(self, df):
        if (df["high"] < df["low"]).any():
            raise ValueError("[Validator] high < low detetado.")

        if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
            raise ValueError("[Validator] open fora dos limites.")

        if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
            raise ValueError("[Validator] close fora dos limites.")

    # ==========================================================
    # 4. Amplitude impossível
    # ==========================================================
    def _check_amplitude(self, df):
        amp = df["high"] - df["low"]
        median_amp = max(amp.median(), 0.0001)

        impossible = amp > median_amp * self.max_amp_factor
        if impossible.any():
            idx = impossible.idxmax()
            ts = df.loc[idx, "timestamp"]
            val = float(amp.loc[idx])
            med = float(median_amp)

            raise ValueError(
                f"[Validator] amplitude impossível em {ts}: "
                f"{val:.4f} vs mediana {med:.4f}"
            )

    # ==========================================================
    # 5. Volume
    # ==========================================================
    def _check_volume(self, df):
        if (df["volume"] < 0).any():
            raise ValueError("[Validator] volume negativo encontrado.")

        # volume zero durante horas pode ser real
        if df["volume"].rolling(200).sum().eq(0).any():
            raise ValueError("[Validator] volume zero prolongado (>=200h).")

    # ==========================================================
    # 6. Comprimento mínimo
    # ==========================================================
    def _check_min_length(self, df):
        if len(df) < self.min_history_points:
            raise ValueError(
                f"[Validator] série demasiado curta ({len(df)}). "
                f"Requer mínimo: {self.min_history_points}"
            )

    # ==========================================================
    # 7. Gaps intradiários — versão final
    # ==========================================================
    def _check_intraday_gaps(self, df):
        ts = df["timestamp"].values

        for i in range(1, len(ts)):
            curr = ts[i]
            prev = ts[i - 1]

            # ignorar break diário
            if pd.Timestamp(curr).date() != pd.Timestamp(prev).date():
                continue

            delta_h = (curr - prev).astype("timedelta64[h]").astype(int)

            if delta_h <= self.max_intraday_gap_hours:
                continue

            raise ValueError(
                f"[Validator] gap intradiário impossível: {delta_h}h "
                f"entre {prev} → {curr}"
            )

    # ==========================================================
    # 8. Estacionaridade opcional
    # ==========================================================
    def _check_stationarity(self, df):
        try:
            p = adfuller(df["close"], autolag="AIC")[1]
        except Exception:
            return
        if p > 0.05:
            raise ValueError("[Validator] série não estacionária (ADF p > 0.05).")

    # ==========================================================
    # 9. Autocorrelação opcional
    # ==========================================================
    def _check_autocorr(self, df):
        if len(df) < 100:
            return
        ac = df["close"].autocorr(lag=1)
        if abs(ac) < 0.01:
            raise ValueError("[Validator] autocorrelação insuficiente (|lag1|<0.01).")

    # ==========================================================
    # 10. Execução principal
    # ==========================================================
    def validate(self, df):

        df = df.copy().reset_index(drop=True)

        self._check_columns(df)
        self._check_nans(df)
        self._check_ohlc_consistency(df)
        self._check_amplitude(df)
        self._check_volume(df)
        self._check_min_length(df)
        self._check_intraday_gaps(df)

        if self.require_stationarity:
            self._check_stationarity(df)

        if self.check_autocorr:
            self._check_autocorr(df)

        return df


# Instância global
validator = Validator()
