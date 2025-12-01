import numpy as np
import pandas as pd


class BacktestMetrics:
    """
    Métricas industriais para avaliação de estratégias.
    Todas matematicamente corretas e compatíveis com o BacktestEngine V3.

    Requer colunas:
        - equity_curve
        - return_net
    """

    # ============================================================
    # MAX DRAWDOWN
    # ============================================================
    @staticmethod
    def max_drawdown(equity: pd.Series):
        """
        Calcula o max drawdown e a série de drawdown.
        """
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min()), dd

    # ============================================================
    # SHARPE RATIO (risk-adjusted return)
    # ============================================================
    @staticmethod
    def sharpe_ratio(returns: pd.Series, freq: int = 252):
        """
        Sharpe com retornos diários anualizados.
        Para crypto podemos usar freq=365.
        """
        mean_ret = returns.mean() * freq
        vol = returns.std() * np.sqrt(freq)

        if vol == 0:
            return 0.0
        return float(mean_ret / vol)

    # ============================================================
    # CALMAR RATIO
    # ============================================================
    @staticmethod
    def calmar_ratio(equity: pd.Series):
        """
        Calmar = retorno total / |max drawdown|
        """
        total_return = float(equity.iloc[-1] - 1)
        mdd, _ = BacktestMetrics.max_drawdown(equity)

        if mdd == 0:
            return np.inf

        return float(total_return / abs(mdd))

    # ============================================================
    # WINRATE
    # ============================================================
    @staticmethod
    def winrate(returns: pd.Series):
        return float((returns > 0).mean())

    # ============================================================
    # VOLATILIDADE ANUALIZADA
    # ============================================================
    @staticmethod
    def volatility(returns: pd.Series, freq=252):
        return float(returns.std() * np.sqrt(freq))

    # ============================================================
    # TOTAL RETURN
    # ============================================================
    @staticmethod
    def total_return(equity: pd.Series):
        return float(equity.iloc[-1] - 1)

    # ============================================================
    # AGRUPAMENTO FINAL DE TODAS AS MÉTRICAS
    # ============================================================
    @staticmethod
    def compute(df: pd.DataFrame):
        if "equity_curve" not in df.columns:
            raise ValueError("BacktestMetrics: falta coluna 'equity_curve'.")
        if "return_net" not in df.columns:
            raise ValueError("BacktestMetrics: falta coluna 'return_net'.")

        equity = df["equity_curve"]
        returns = df["return_net"]

        mdd, dd_series = BacktestMetrics.max_drawdown(equity)

        metrics = {
            "sharpe": BacktestMetrics.sharpe_ratio(returns),
            "calmar": BacktestMetrics.calmar_ratio(equity),
            "winrate": BacktestMetrics.winrate(returns),
            "max_drawdown": float(mdd),
            "volatility": BacktestMetrics.volatility(returns),
            "total_return": BacktestMetrics.total_return(equity),
        }

        return metrics
