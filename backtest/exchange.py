from __future__ import annotations
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from .order import Order
from .position import Position


@dataclass
class Fill:
    """
    Representa uma execução (fill) de uma ordem.
    """
    order_id: int
    price: float
    size: float
    side: int    # 1 long, -1 short


class Exchange:
    """
    Exchange V1 — deterministic & industrial matching engine.

    Funções principais:
        - receber ordens
        - executar ordens conforme preço de mercado
        - aplicar slippage
        - gerar fills
        - atualizar posição
        - gerir stop-loss, take-profit e OCO
        - cancelar ordens expiradas ou invalidadas

    Regras:
        - Apenas preços OHLC são usados para execução.
        - Slippage é opcional e aplicado a market orders.
        - As ordens são processadas sempre na mesma sequência:
            1. Market → pre-bar close
            2. Stop trigger
            3. Limit fills
            4. SL / TP
    """

    def __init__(
        self,
        slippage: float | Callable[[float], float] = 0.0
    ):
        """
        slippage:
            - número → slippage absoluto (p.ex. 0.01)
            - função → slippage dinâmica (price -> slip)
        """
        self.slippage = slippage
        self.orders: List[Order] = []    # ordens pendentes
        self.position = Position()       # posição ativa
        self.order_counter = 0

    # ============================================================
    # Utils
    # ============================================================
    def _apply_slippage(self, price: float) -> float:
        if callable(self.slippage):
            return price + self.slippage(price)
        return price + self.slippage

    # ============================================================
    # Gestão de ordens
    # ============================================================
    def submit(self, order: Order):
        """
        Regista nova ordem com ID sequencial.
        """
        self.order_counter += 1
        order.id = self.order_counter
        self.orders.append(order)

    def cancel(self, order_id: int):
        self.orders = [o for o in self.orders if o.id != order_id]

    def cancel_all(self):
        self.orders = []

    # ============================================================
    # Execução principal
    # ============================================================
    def process_bar(
        self,
        timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float
    ):
        """
        Processa 1 vela.

        Ordem de execução:
            1. Market orders → executam no open da vela
            2. Stop orders → verificam triggers no range
            3. Stop-limit → convertidas conforme trigger
            4. Limit orders → executam se houver liquidez
            5. Stop-Loss / Take-Profit ligados à posição
        """

        # ==========================================
        # 1. EXECUTAR MARKET ORDERS
        # ==========================================
        fills = []
        remaining_orders = []

        for o in self.orders:
            if o.type == "market":
                px = self._apply_slippage(open_price)
                fill = Fill(o.id, px, o.size, o.side)
                fills.append(fill)
                self._apply_fill(fill)
            else:
                remaining_orders.append(o)

        self.orders = remaining_orders

        # ==========================================
        # 2. EXECUTAR STOP ORDERS (trigger)
        # ==========================================
        triggered = []
        remaining_orders = []

        for o in self.orders:
            if o.type == "stop":
                if (o.side == 1 and high >= o.stop_price) or \
                   (o.side == -1 and low <= o.stop_price):
                    # Triggered → vira market com slippage
                    px = self._apply_slippage(open_price)
                    fill = Fill(o.id, px, o.size, o.side)
                    fills.append(fill)
                    self._apply_fill(fill)
                else:
                    remaining_orders.append(o)
            else:
                remaining_orders.append(o)

        self.orders = remaining_orders

        # ==========================================
        # 3. STOP-LIMIT ORDERS
        # ==========================================
        converted_orders = []
        remaining_orders = []

        for o in self.orders:
            if o.type == "stop_limit":
                triggered = (
                    (o.side == 1 and high >= o.stop_price) or
                    (o.side == -1 and low <= o.stop_price)
                )
                if triggered:
                    # converte para uma ordem limit clássica
                    o.type = "limit"
                    converted_orders.append(o)
                else:
                    remaining_orders.append(o)
            else:
                remaining_orders.append(o)

        # reintroduz converted limit orders
        self.orders = remaining_orders + converted_orders

        # ==========================================
        # 4. LIMIT ORDERS EXECUTION
        # ==========================================
        remaining_orders = []

        for o in self.orders:
            if o.type == "limit":
                # Buy limit fills if market touched <= limit
                # Sell limit fills if market touched >= limit
                executed = (
                    (o.side == 1 and low <= o.limit_price) or
                    (o.side == -1 and high >= o.limit_price)
                )

                if executed:
                    px = o.limit_price
                    fill = Fill(o.id, px, o.size, o.side)
                    fills.append(fill)
                    self._apply_fill(fill)
                else:
                    remaining_orders.append(o)

            else:
                remaining_orders.append(o)

        self.orders = remaining_orders

        # ==========================================
        # 5. STOP LOSS / TAKE PROFIT (da posição)
        # ==========================================
        self._process_sl_tp(low, high)

        # Atualizar posição (unrealized / mfe / mae)
        self.position.update_unrealized(close)

        return fills

    # ============================================================
    # Aplicar fill na posição
    # ============================================================
    def _apply_fill(self, fill: Fill):
        """
        Integra execução com a posição.
        """
        side = fill.side
        price = fill.price
        size = fill.size

        if self.position.is_flat():
            self.position.open(side, size, price)
            return

        # Se side coincide com posição → aumentar
        if self.position.side == side:
            self.position.increase(size, price)
            return

        # Caso contrário → reduzir ou flipar
        if size < self.position.size:
            self.position.reduce(size, price)
            return

        # size >= posição → fecha e vira
        self.position.flip(side, size, price)

    # ============================================================
    # STOP LOSS & TAKE PROFIT nativos
    # ============================================================
    def _process_sl_tp(self, low: float, high: float):
        pos = self.position

        if pos.is_flat():
            return

        # SL / TP só se definidos
        sl = getattr(pos, "stop_loss", None)
        tp = getattr(pos, "take_profit", None)

        # STOP LOSS
        if sl is not None:
            if pos.side == 1 and low <= sl:
                pos.close_all(sl)
            elif pos.side == -1 and high >= sl:
                pos.close_all(sl)

        # TAKE PROFIT
        if tp is not None:
            if pos.side == 1 and high >= tp:
                pos.close_all(tp)
            elif pos.side == -1 and low <= tp:
                pos.close_all(tp)
