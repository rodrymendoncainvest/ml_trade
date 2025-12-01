import numpy as np
import pandas as pd


class FeatureGenerator:
    """
    Feature Generator V3 Industrial.
    - Indicadores robustos, matematicamente seguros.
    - Proteções anti-NaN em todas as operações.
    - Price Action avançado.
    - Momentum, tendência, volume.
    - Volatilidade moderna (GK, Parkinson, RS).
    - Bollinger, Keltner, Squeeze (Robusto).
    - Patterns de velas (binários).
    - VWAP + VWAP bands.
    - Regime técnico (trend/vol/mom).
    """

    # ============================================================
    # AUXILIARES SEGUROS
    # ============================================================

    @staticmethod
    def _ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series, period):
        return series.rolling(period).mean()

    @staticmethod
    def _true_range(df):
        prev_close = df["close"].shift(1)
        ranges = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1)
        return ranges.max(axis=1)

    # ============================================================
    # PRICE ACTION (SEGURO)
    # ============================================================

    def _price_action(self, df):
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        body = (close - open_).abs()
        direction = np.sign(close - open_).fillna(0)

        total_range = (high - low).replace(0, 1e-9)

        upper_wick = (high - np.maximum(open_, close))
        lower_wick = (np.minimum(open_, close) - low)

        body_ratio = (body / total_range).fillna(0)
        range_rel = total_range / total_range.rolling(20).mean()
        range_rel = range_rel.replace([np.inf, -np.inf], 0)

        return body, upper_wick, lower_wick, body_ratio, direction, range_rel

    # ============================================================
    # VELAS / PATTERNS (BINÁRIOS, PARCIAIS)
    # ============================================================

    def _patterns(self, df):
        body = (df["close"] - df["open"]).abs()
        total = (df["high"] - df["low"]).replace(0, 1e-9)

        small_body = (body / total) < 0.2
        long_body = (body / total) > 0.7

        bullish = df["close"] > df["open"]
        bearish = df["close"] < df["open"]

        hammer = (small_body & (df["low"] < df["open"]) &
                  ((df["open"] - df["low"]) > 2 * body)).astype(int)

        engulfing = (
            (bullish & (df["close"] > df["open"].shift(1))) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)

        return small_body.astype(int), long_body.astype(int), hammer, engulfing

    # ============================================================
    # MOMENTUM / TREND
    # ============================================================

    def _momentum(self, close, period=10):
        return (close - close.shift(period)).fillna(0)

    def _trend(self, df):
        close = df["close"]
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        macd_line = self._ema(close, 12) - self._ema(close, 26)
        return ema20 - ema50, macd_line

    # ============================================================
    # OSCILADORES
    # ============================================================

    def _rsi(self, close, period=14):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down.replace(0, 1e-9)
        return (100 - (100 / (1 + rs))).fillna(50)

    def _stochastic(self, df, period=14, smooth=3):
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-9)
        d = k.rolling(smooth).mean()
        return k.fillna(50), d.fillna(50)

    # ============================================================
    # VOLATILIDADE MODERNA
    # ============================================================

    def _parkinson(self, df):
        v = (np.log(df["high"] / df["low"]) ** 2).replace([np.inf, -np.inf], 0)
        return v.rolling(20).mean()

    def _garman_klass(self, df):
        hl = np.log(df["high"] / df["low"]) ** 2
        oc = 0.5 * (np.log(df["close"] / df["open"]) ** 2)
        vol = (hl - oc).rolling(20).mean()
        return vol.replace([np.inf, -np.inf], 0)

    def _rogers_satchell(self, df):
        term1 = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"])
        term2 = np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
        vol = (term1 + term2).rolling(20).mean()
        return vol.replace([np.inf, -np.inf], 0)

    # ============================================================
    # BOLLINGER, KELTNER, SQUEEZE
    # ============================================================

    def _bollinger(self, close, period=20, std_mult=2):
        mid = self._sma(close, period)
        std = close.rolling(period).std()
        upper = mid + std * std_mult
        lower = mid - std * std_mult
        return mid, upper, lower

    def _keltner(self, df, period=20):
        tr = self._true_range(df)
        mid = self._ema(df["close"], period)
        return mid, mid + 1.5 * tr.rolling(period).mean(), mid - 1.5 * tr.rolling(period).mean()

    def _squeeze(self, close, bb_upper, bb_lower, kc_upper, kc_lower):
        return ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)

    # ============================================================
    # VOLUME
    # ============================================================

    def _obv(self, df):
        return (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    def _vwap(self, df):
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum().replace(0, 1e-9)
        cum_pv = (typical * df["volume"]).cumsum()
        return (cum_pv / cum_vol).fillna(method="bfill").fillna(method="ffill")

    # ============================================================
    # REGIME
    # ============================================================

    def _regime(self, df):
        close = df["close"]
        vol = close.pct_change().rolling(20).std().fillna(0)
        trend = self._ema(close, 20) - self._ema(close, 50)
        mom = close.diff(5).fillna(0)
        return trend, vol, mom

    # ============================================================
    # GERAÇÃO PRINCIPAL
    # ============================================================

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # PRICE ACTION
        body, uw, lw, br, direction, rrel = self._price_action(df)
        df["body"] = body
        df["upper_wick"] = uw
        df["lower_wick"] = lw
        df["body_ratio"] = br
        df["direction"] = direction
        df["range_rel"] = rrel

        # PATTERNS
        small, longb, hammer, engulf = self._patterns(df)
        df["pattern_small"] = small
        df["pattern_long"] = longb
        df["pattern_hammer"] = hammer
        df["pattern_engulf"] = engulf

        # OSCILADORES
        df["rsi_14"] = self._rsi(df["close"])
        k, d = self._stochastic(df)
        df["stoch_k"] = k
        df["stoch_d"] = d

        # TREND / MOMENTUM
        trend, macd_line = self._trend(df)
        df["trend"] = trend
        df["macd_line"] = macd_line
        df["mom_10"] = self._momentum(df["close"], 10)

        # VOLATILIDADE MODERNA
        df["park_vol"] = self._parkinson(df)
        df["gk_vol"] = self._garman_klass(df)
        df["rs_vol"] = self._rogers_satchell(df)

        # BOLLINGER / KELTNER / SQUEEZE
        mid, bb_u, bb_l = self._bollinger(df["close"])
        kmid, ku, kl = self._keltner(df)
        df["bb_mid"] = mid
        df["bb_upper"] = bb_u
        df["bb_lower"] = bb_l
        df["kc_upper"] = ku
        df["kc_lower"] = kl
        df["squeeze_on"] = self._squeeze(df["close"], bb_u, bb_l, ku, kl)

        # VOLUME
        df["obv"] = self._obv(df)
        df["vwap"] = self._vwap(df)

        # REGIME
        tr, vol, mom = self._regime(df)
        df["regime_trend"] = tr
        df["regime_vol"] = vol
        df["regime_mom"] = mom

        # CLEANUP
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)

        return df
