from __future__ import annotations
import pandas as pd
from typing import Callable, Optional, Dict, Any

from .exchange import Exchange
from .order import Order
from .position import Position


class BacktestEngine:
    """
    BacktestEngine V1 — Industrial & Determinística

    Responsabilidades:
    ------------------------------------------------------
    - Correr barra-a-barra (process_bar)
    - Enviar ordens das estratégias para a Exchange
    - Atualizar posição após cada vela
    - Guardar todos os fills e todas as ordens executadas
    - Suportar qualquer estratégia baseada em sinais

    Suporta:
    ------------------------------------------------------
    ✔ Sinais de estratégia estilo "signal = +1 / -1 / 0"
    ✔ Estratégias complexas com múltiplas ordens
    ✔ Acesso direto à Exchange
    ✔ Posição sempre consistente
    ✔ SL/TP nativos (via Position)
    ✔ Histórico completo de operações
    """

    def __init__(
        self,
        strategy_func: Callable,
        fee: float = 0.0004,
        slippage: float = 0.0,
    ):
        """
        strategy_func: função que recebe (engine, i) e que devolve:
            - ordem(s) ou
            - um sinal (1, -1, 0) ou
            - nada (None)

        fee      → custo de transação
        slippage → slippage aplicado pela Exchange
        """
        self.strategy_func = strategy_func
        self.exchange = Exchange(slippage=slippage)
        self.fee = fee

        self.fills_log = []       # histórico de fills
        self.orders_log = []      # histórico de ordens emitidas

    # ============================================================
    #  SUBMETER ORDENS
    # ============================================================
    def submit_order(self, order: Order):
        """Adiciona ordem à Exchange e guarda no log."""
        self.exchange.submit(order)
        self.orders_log.append(order)

    # ============================================================
    #  CONSUMIR SINAL SIMPLES
    # ============================================================
    def submit_signal(self, signal: int, size: float = 1.0):
        """
        Converte sinais discretos (+1 long, -1 short, 0 close)
        em ordens de mercado diretas.
        """
        pos = self.exchange.position

        if signal == 1:
            if pos.side == 1:
                return  # já estamos long
            if pos.side == -1:
                # fechar short antes de virar long
                self.submit_order(Order(type="market", side=1, size=pos.size * 2))
            else:
                self.submit_order(Order(type="market", side=1, size=size))

        elif signal == -1:
            if pos.side == -1:
                return
            if pos.side == 1:
                self.submit_order(Order(type="market", side=-1, size=pos.size * 2))
            else:
                self.submit_order(Order(type="market", side=-1, size=size))

        elif signal == 0:
            if not pos.is_flat():
                # fechar qualquer posição
                self.submit_order(Order(type="market", side=-pos.side, size=pos.size))

    # ============================================================
    #  LOOP PRINCIPAL — BAR-BY-BAR
    # ============================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corre o motor completo e devolve um DataFrame com:

            - equity_curve
            - return_net
            - return_strategy
            - position_size
            - position_side
            - entry_price
            - MFE / MAE
            - sinais / fills

        """

        # Criar colunas de tracking
        df = df.copy().reset_index(drop=True)
        df["signal"] = 0
        df["position_side"] = 0
        df["position_size"] = 0.0
        df["entry_price"] = 0.0
        df["unrealized"] = 0.0
        df["mfe"] = 0.0
        df["mae"] = 0.0

        position = self.exchange.position

        for i in range(1, len(df)):
            row = df.iloc[i]
            timestamp = row["timestamp"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # -----------------------------------------
            # 1. Estratégia gera ordens ou sinal
            # -----------------------------------------
            cmd = self.strategy_func(self, i)

            if isinstance(cmd, Order):
                self.submit_order(cmd)

            elif isinstance(cmd, (list, tuple)):
                for o_ in cmd:
                    self.submit_order(o_)

            elif isinstance(cmd, int):  # sinal discreto
                df.at[i, "signal"] = cmd
                self.submit_signal(cmd)

            # -----------------------------------------
            # 2. Exchange executa ordens nesta vela
            # -----------------------------------------
            fills = self.exchange.process_bar(
                timestamp=timestamp,
                open_price=o,
                high=h,
                low=l,
                close=c
            )

            # log fills
            for f in fills:
                self.fills_log.append(f)

            # -----------------------------------------
            # 3. Tracking da posição
            # -----------------------------------------
            df.at[i, "position_side"] = position.side
            df.at[i, "position_size"] = position.size
            df.at[i, "entry_price"] = position.entry_price
            df.at[i, "unrealized"] = position.unrealized
            df.at[i, "mfe"] = position.mfe
            df.at[i, "mae"] = position.mae

        # -----------------------------------------------------------
        # 4. Construir curva de equity (risk-free e sem fees na posição)
        # -----------------------------------------------------------
        df["return_mkt"] = df["close"].pct_change().fillna(0)
        df["return_strategy"] = df["signal"].shift(1) * df["return_mkt"]
        df["return_net"] = df["return_strategy"]  # fees já aplicadas na exchange

        df["equity_curve"] = (1 + df["return_net"]).cumprod()

        return df
