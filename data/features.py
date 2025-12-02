import numpy as np
import pandas as pd


class FeatureGenerator:
    """
    Feature Generator V4 Industrial.
    Robusto, seguro, sem warnings, sem NaNs, sem infinities.
    """

    # ============================================================
    # AUXILIARES SEGUROS
    # ============================================================

    @staticmethod
    def _safe_div(a, b):
        b_safe = b.replace(0, 1e-9)
        out = a / b_safe
        return out.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def _safe_log(a):
        a_safe = a.replace(0, np.nan)
        out = np.log(a_safe)
        return out.replace([np.inf, -np.inf], 0).fillna(0)

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
    # PRICE ACTION SEGURO
    # ============================================================

    def _price_action(self, df):
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        body = (c - o).abs()
        direction = np.sign(c - o).fillna(0)

        total_range = (h - l).replace(0, 1e-9)

        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l

        body_ratio = self._safe_div(body, total_range)

        mean_range = total_range.rolling(20).mean().replace(0, 1e-9)
        range_rel = self._safe_div(total_range, mean_range)

        return body, upper, lower, body_ratio, direction, range_rel

    # ============================================================
    # CANDLE PATTERNS (SEGUROS)
    # ============================================================

    def _patterns(self, df):
        body = (df["close"] - df["open"]).abs()
        total = (df["high"] - df["low"]).replace(0, 1e-9)

        ratio = self._safe_div(body, total)

        small_body = (ratio < 0.2).astype(int)
        long_body = (ratio > 0.7).astype(int)

        bullish = (df["close"] > df["open"]).astype(int)
        bearish = (df["close"] < df["open"]).astype(int)

        # Hammer robusto
        hammer = (
            (ratio < 0.2) &
            ((df["open"] - df["low"]) > 2 * body)
        ).astype(int)

        # Engulfing robusto
        engulf = (
            (df["close"] > df["open"]) &
            (df["open"] < df["close"].shift(1)) &
            (df["close"] > df["open"].shift(1))
        ).astype(int)

        return small_body, long_body, hammer, engulf

    # ============================================================
    # OSCILADORES
    # ============================================================

    def _rsi(self, close, period=14):
        delta = close.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)

        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean().replace(0, 1e-9)

        rs = self._safe_div(ma_up, ma_down)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _stochastic(self, df, period=14, smooth=3):
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max().replace(0, 1e-9)

        k = 100 * self._safe_div(df["close"] - low_min, high_max - low_min)
        d = k.rolling(smooth).mean()

        return k.fillna(50), d.fillna(50)

    # ============================================================
    # TREND & MOMENTUM
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
    # VOLATILIDADES (Blindadas)
    # ============================================================

    def _parkinson(self, df):
        out = (self._safe_log(df["high"] / df["low"]) ** 2).rolling(20).mean()
        return out.fillna(0)

    def _garman_klass(self, df):
        hl = (self._safe_log(df["high"] / df["low"]) ** 2)
        oc = 0.5 * (self._safe_log(df["close"] / df["open"]) ** 2)
        vol = (hl - oc).rolling(20).mean()
        return vol.fillna(0)

    def _rogers_satchell(self, df):
        term1 = self._safe_log(df["high"] / df["close"]) * self._safe_log(df["high"] / df["open"])
        term2 = self._safe_log(df["low"] / df["close"]) * self._safe_log(df["low"] / df["open"])
        vol = (term1 + term2).rolling(20).mean()
        return vol.replace([np.inf, -np.inf], 0).fillna(0)

    # ============================================================
    # BOLLINGER / KELTNER / SQUEEZE
    # ============================================================

    def _bollinger(self, close, period=20, std_mult=2):
        mid = self._sma(close, period)
        std = close.rolling(period).std().fillna(0)
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        return mid, upper, lower

    def _keltner(self, df, period=20):
        tr = self._true_range(df)
        mid = self._ema(df["close"], period)
        atr = tr.rolling(period).mean().replace(0, 1e-9)
        upper = mid + 1.5 * atr
        lower = mid - 1.5 * atr
        return mid, upper, lower

    def _squeeze(self, bb_u, bb_l, kc_u, kc_l):
        return ((bb_l > kc_l) & (bb_u < kc_u)).astype(int)

    # ============================================================
    # VOLUME & VWAP (blindados)
    # ============================================================

    def _obv(self, df):
        sign = np.sign(df["close"].diff()).fillna(0)
        return (sign * df["volume"]).cumsum().fillna(0)

    def _vwap(self, df):
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].replace(0, 1e-9)

        cum_vol = vol.cumsum().replace(0, 1e-9)
        cum_pv = (typical * vol).cumsum()

        return self._safe_div(cum_pv, cum_vol)

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
        body, uw, lw, br, direction, range_rel = self._price_action(df)
        df["body"] = body
        df["upper_wick"] = uw
        df["lower_wick"] = lw
        df["body_ratio"] = br
        df["direction"] = direction
        df["range_rel"] = range_rel

        # PATTERNS
        p_small, p_long, hammer, engulf = self._patterns(df)
        df["pattern_small"] = p_small
        df["pattern_long"] = p_long
        df["pattern_hammer"] = hammer
        df["pattern_engulf"] = engulf

        # OSCILADORES
        df["rsi_14"] = self._rsi(df["close"])
        k, d = self._stochastic(df)
        df["stoch_k"] = k
        df["stoch_d"] = d

        # TREND & MOMENTUM
        trend, macd_line = self._trend(df)
        df["trend"] = trend
        df["macd_line"] = macd_line
        df["mom_10"] = self._momentum(df["close"], 10)

        # VOLATILIDADE
        df["park_vol"] = self._parkinson(df)
        df["gk_vol"] = self._garman_klass(df)
        df["rs_vol"] = self._rogers_satchell(df)

        # BOLLINGER + KELTNER + SQUEEZE
        mid, bb_u, bb_l = self._bollinger(df["close"])
        kmid, ku, kl = self._keltner(df)
        df["bb_mid"] = mid
        df["bb_upper"] = bb_u
        df["bb_lower"] = bb_l
        df["kc_upper"] = ku
        df["kc_lower"] = kl
        df["squeeze_on"] = self._squeeze(bb_u, bb_l, ku, kl)

        # VOLUME + VWAP
        df["obv"] = self._obv(df)
        df["vwap"] = self._vwap(df)

        # REGIME
        tr, vol, mom = self._regime(df)
        df["regime_trend"] = tr
        df["regime_vol"] = vol
        df["regime_mom"] = mom

        # FINAL CLEANUP (sem drop maciço)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(0)

        return df
