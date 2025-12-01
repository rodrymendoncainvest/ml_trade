import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()


def sharpe_ratio(returns: pd.Series, risk_free=0.0):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / returns.std()


def annualize_return(total_return: float, periods: int, freq="1h"):
    if freq == "1h":
        annual_factor = 24 * 365
    elif freq == "1d":
        annual_factor = 365
    else:
        annual_factor = periods
    return (1 + total_return) ** (annual_factor / periods) - 1


def compute_metrics(result: pd.DataFrame):
    """
    Requer resultado do engine:
      timestamp, close, signal, pnl, equity
    """
    eq = result["equity"]
    ret = result["pnl"]

    total_ret = eq.iloc[-1] / max(1e-9, abs(eq.iloc[0])) if eq.iloc[0] != 0 else 0.0

    metrics = {
        "total_return": float(total_ret),
        "max_drawdown": float(max_drawdown(eq)),
        "sharpe_ratio": float(sharpe_ratio(ret)),
        "volatility": float(ret.std()),
        "mean_pnl": float(ret.mean()),
        "winrate": float((ret > 0).mean()),
        "annualized_return": float(annualize_return(total_ret, len(result))),
    }
    return metrics
