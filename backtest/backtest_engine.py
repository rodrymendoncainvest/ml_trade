import numpy as np
import pandas as pd


class BacktestEngine:
    """
    BacktestEngine V4 — Industrial, Determinístico e Modular.

    Melhorias face à V3:
    -----------------------------------
    ✔ suporte para estratégias externas (strategy.generate_signals)
    ✔ API unificada: run(), apply_strategy()
    ✔ SL / TP profissionais aplicados candle-by-candle
    ✔ position sizing fixo ou variável
    ✔ cálculo vetorizado sempre que possível
    ✔ compatível com sinais vindos de:
        - preço (baseline)
        - previsões ML
        - estratégia híbrida externa
    ✔ nunca altera colunas existentes — só adiciona novas
    """

    def __init__(
        self,
        fee: float = 0.0004,
        threshold: float = 0.001,
        position_size: float = 1.0,
    ):
        """
        fee: custo por mudança de posição
        threshold: limite mínimo para gerar long/short nos métodos internos
        position_size: tamanho da posição (1.0 = 100%)
        """
        self.fee = fee
        self.threshold = threshold
        self.position_size = position_size

    # ================================================================
    # 1) SINAIS A PARTIR DO CLOSE (baseline simples)
    # ================================================================
    def build_signals_from_close(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("BacktestEngine: falta coluna 'close'.")

        ret = df["close"].pct_change().fillna(0)

        sig = np.where(
            ret > self.threshold, 1,
            np.where(ret < -self.threshold, -1, 0)
        )

        return pd.Series(sig, index=df.index, name="signal")

    # ================================================================
    # 2) SINAIS A PARTIR DO MODELO (previsões ML)
    # ================================================================
    def build_signals_from_prediction(self, df: pd.DataFrame) -> pd.Series:
        if "prediction" not in df.columns:
            raise ValueError("BacktestEngine: falta coluna 'prediction'.")

        pred = df["prediction"]

        sig = np.where(
            pred > self.threshold, 1,
            np.where(pred < -self.threshold, -1, 0)
        )

        return pd.Series(sig, index=df.index, name="signal")

    # ================================================================
    # 3) APLICAR UMA ESTRATÉGIA EXTERNA (HybridMLStrategy)
    # ================================================================
    def apply_strategy(self, df: pd.DataFrame, strategy) -> pd.Series:
        """
        strategy: objeto com método generate_signals(df) -> Series
        """
        if not hasattr(strategy, "generate_signals"):
            raise ValueError("Estratégia não contém método generate_signals(df).")

        sig = strategy.generate_signals(df)

        if not isinstance(sig, pd.Series):
            raise ValueError("generate_signals deve retornar pandas.Series.")

        return sig.rename("signal")

    # ================================================================
    # 4) EXECUÇÃO COMPLETA DO BACKTEST
    # ================================================================
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        stop_loss: float = None,
        take_profit: float = None,
    ):
        """
        df           : OHLCV + features
        signals      : Série (1, -1, 0)
        stop_loss    : percent (ex: 0.01 = -1%)
        take_profit  : percent (ex: 0.02 = +2%)
        """

        df = df.copy()

        # -------------------------------------
        # garantir alinhamento
        # -------------------------------------
        signals = signals.reindex(df.index).fillna(0).astype(int)
        df["signal"] = signals

        # -------------------------------------
        # retorno do mercado
        # -------------------------------------
        df["return_mkt"] = df["close"].pct_change().fillna(0)

        # -------------------------------------
        # retorno bruto da estratégia
        # -------------------------------------
        df["return_strategy"] = df["signal"].shift(1) * df["return_mkt"]
        df["return_strategy"] = df["return_strategy"].fillna(0)

        # -------------------------------------
        # aplicar SL / TP — candle by candle
        # -------------------------------------
        if stop_loss or take_profit:
            df["return_strategy"] = self._apply_sl_tp(
                df["return_strategy"],
                df["signal"],
                stop_loss,
                take_profit,
            )

        # -------------------------------------
        # aplicar fees
        # -------------------------------------
        df["trade"] = (df["signal"] != df["signal"].shift(1)).astype(int)
        df["fee_cost"] = df["trade"] * self.fee

        # -------------------------------------
        # retorno líquido
        # -------------------------------------
        df["return_net"] = (
            df["return_strategy"] - df["fee_cost"]
        ) * self.position_size

        # -------------------------------------
        # curva de equity
        # -------------------------------------
        df["equity_curve"] = (1 + df["return_net"]).cumprod()

        return df

    # ================================================================
    # 5) STOP-LOSS / TAKE-PROFIT PROFISSIONAIS
    # ================================================================
    def _apply_sl_tp(self, returns, signals, sl, tp):
        """
        sl / tp aplicados ao retorno bruto da posição antes de fees.
        returns = retorno da posição baseada no sinal anterior.
        """

        returns = returns.copy()

        if sl is None and tp is None:
            return returns

        for i in range(1, len(returns)):
            pos = signals.iloc[i - 1]

            if pos == 0:
                continue

            r = returns.iloc[i]

            # long position
            if pos == 1:
                if sl is not None and r < -sl:
                    returns.iloc[i] = -sl
                elif tp is not None and r > tp:
                    returns.iloc[i] = tp

            # short position
            if pos == -1:
                if sl is not None and r > sl:
                    returns.iloc[i] = sl
                elif tp is not None and r < -tp:
                    returns.iloc[i] = -tp

        return returns
