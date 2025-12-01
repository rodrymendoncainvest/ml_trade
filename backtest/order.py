import enum
from dataclasses import dataclass, field
from typing import Optional


class OrderSide(enum.Enum):
    BUY = 1
    SELL = -1


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"              # stop-market
    STOP_LIMIT = "stop_limit"  # stop-limit
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"


class OrderStatus(enum.Enum):
    NEW = "new"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Ordem profissional, totalmente determinística e compatível com o motor de execução.
    Implementa:
        - Market
        - Limit
        - Stop
        - Stop-Limit
        - SL / TP
        - OCO (através do campo oco_id)
    """

    id: int
    side: OrderSide
    type: OrderType
    size: float

    # preços
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # gestão avançada
    oco_id: Optional[int] = None  # One-Cancels-Other

    # estado
    status: OrderStatus = OrderStatus.NEW
    filled_size: float = 0.0
    avg_fill_price: float = 0.0

    # controlo interno
    created_at: int = 0
    activated_at: Optional[int] = None
    filled_at: Optional[int] = None

    def activate(self, ts_index: int):
        """
        Ordem torna-se ativa quando:
        - market: imediatamente
        - limit/stop: motor decide
        """
        if self.status == OrderStatus.NEW:
            self.status = OrderStatus.ACTIVE
            self.activated_at = ts_index

    def cancel(self):
        if self.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            self.status = OrderStatus.CANCELLED

    def is_active(self):
        return self.status == OrderStatus.ACTIVE

    def is_new(self):
        return self.status == OrderStatus.NEW

    def is_filled(self):
        return self.status == OrderStatus.FILLED

    def can_fill(self):
        return self.status in (OrderStatus.ACTIVE, OrderStatus.NEW)

    def mark_filled(self, price: float, ts_index: int):
        self.filled_size = self.size
        self.avg_fill_price = price
        self.status = OrderStatus.FILLED
        self.filled_at = ts_index
