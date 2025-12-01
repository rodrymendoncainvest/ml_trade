import numpy as np
import pandas as pd


# ============================================================
# Segurança matemática
# ============================================================

def safe_div(a, b):
    """Divisão segura que evita ZeroDivisionError."""
    if isinstance(b, (int, float)) and b == 0:
        return 0.0
    if isinstance(b, np.ndarray):
        b = np.where(b == 0, 1e-8, b)
    return a / b


# ============================================================
# Rolling helpers
# ============================================================

def rolling_mean(series: pd.Series, window: int):
    return series.rolling(window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int):
    return series.rolling(window, min_periods=1).std()


def rolling_min(series: pd.Series, window: int):
    return series.rolling(window, min_periods=1).min()


def rolling_max(series: pd.Series, window: int):
    return series.rolling(window, min_periods=1).max()


# ============================================================
# Estatísticas rápidas
# ============================================================

def zscore(series: pd.Series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series * 0
    return (series - mean) / std


def pct_change(series: pd.Series):
    return series.pct_change().fillna(0)


def log_change(series: pd.Series):
    return np.log(series / series.shift(1)).fillna(0)


# ============================================================
# Verificações básicas
# ============================================================

def is_monotonic(series: pd.Series) -> bool:
    return series.is_monotonic_increasing


def count_missing(series: pd.Series) -> int:
    return series.isna().sum()


# ============================================================
# Normalizações simples
# ============================================================

def minmax(series: pd.Series):
    min_v = series.min()
    max_v = series.max()
    if max_v - min_v == 0:
        return series * 0
    return (series - min_v) / (max_v - min_v)


def normalize(series: pd.Series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series * 0
    return (series - mean) / std


# ============================================================
# Conversores auxiliares
# ============================================================

def to_numpy(array_like):
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32)
    return np.array(array_like, dtype=np.float32)
