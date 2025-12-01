import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Motor de backtest determinístico industrial.

    Responsabilidades:
    - executar sinais (long, short, flat)
    - controlar posição, PnL, slippage e custos
    - evitar qualquer lookahead
    - garantir que cada trade obedece às regras
    """

    def __init__(
        self,
        slippage=0.0,
        fee=0.0,
        contract_size=1.0,
    ):
        """
        slippage: custo adicional por trade (ex: 0.0005 = 0.05%)
        fee: comissão por transação
        contract_size: multiplicador (ex: crypto = 1, futures = 10/100)
        """
        self.slippage = slippage
        self.fee = fee
        self.contract_size = contract_size

    # ======================================================
    # EXECUÇÃO PRINCIPAL
    # ======================================================
    def run(self, df: pd.DataFrame, signals: pd.Series):
        """
        df: DataFrame com OHLC (close usado para execução)
        signals: Série com valores {-1,0,1}
        """

        df = df.copy().reset_index(drop=True)
        assert "close" in df.columns, "BacktestEngine: falta coluna close."

        prices = df["close"].values
        sig = signals.values.astype(int)

        n = len(df)

        position = 0
        entry_price = 0
        pnl = np.zeros(n)
        equity = np.zeros(n)

        for i in range(1, n):
            prev_pos = position
            new_pos = sig[i - 1]    # EXECUÇÃO T+1 obrigatória

            # Se mudou de posição → fechar ou abrir
            if new_pos != prev_pos:
                # fechar posição anterior
                if prev_pos != 0:
                    pnl[i] += prev_pos * (prices[i] - entry_price) * self.contract_size

                # abrir nova posição
                if new_pos != 0:
                    entry_price = prices[i] * (1 + self.slippage * np.sign(new_pos))
                    pnl[i] -= self.fee * abs(new_pos)

            # manter posição anterior
            elif prev_pos != 0:
                pnl[i] += prev_pos * (prices[i] - prices[i - 1]) * self.contract_size

            position = new_pos
            equity[i] = equity[i - 1] + pnl[i]

        result = pd.DataFrame({
            "timestamp": df["timestamp"],
            "close": prices,
            "signal": sig,
            "pnl": pnl,
            "equity": equity,
        })

        return result
