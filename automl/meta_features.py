# automl/meta_features.py
import numpy as np
import pandas as pd


class MetaFeatures:
    """
    MetaFeatures V1 — Normaliza métricas de avaliação para AutoML.
    Todas as métricas são transformadas para:
        +1 = melhor
        0  = neutro
        -1 = pior
    """

    @staticmethod
    def normalize_sharpe(sharpe):
        # Sharpe intraday anualizado (1H) — freq = 24*252 = 6048
        if np.isnan(sharpe):
            return -1
        return np.tanh(sharpe / 3)

    @staticmethod
    def normalize_drawdown(mdd):
        # drawdown negativo: quanto mais perto de 0, melhor
        if mdd >= 0:
            return -1
        return np.tanh((-mdd) / 0.5)

    @staticmethod
    def normalize_total_return(ret):
        # retorno > 0 é bom, retorno negativo é mau
        return np.tanh(ret / 0.20)

    @staticmethod
    def normalize_volatility(vol):
        # queremos menos volatilidade → mais estável
        if vol <= 0:
            return 1
        return np.tanh(1 / (1 + vol))

    @staticmethod
    def aggregate(metrics: dict):
        """
        Combina todas as métricas numa única pontuação H4.
        H4 = média simples dos fatores normalizados.
        """

        s = MetaFeatures.normalize_sharpe(metrics["sharpe"])
        d = MetaFeatures.normalize_drawdown(metrics["max_drawdown"])
        r = MetaFeatures.normalize_total_return(metrics["total_return"])
        v = MetaFeatures.normalize_volatility(metrics["volatility"])

        score = np.mean([s, d, r, v])

        return {
            "norm_sharpe": float(s),
            "norm_drawdown": float(d),
            "norm_total_return": float(r),
            "norm_volatility": float(v),
            "h4_score": float(score),
        }
