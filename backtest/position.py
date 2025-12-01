from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """
    Position V1 — Industrial & determinística

    Este objeto representa SEMPRE a posição ativa do sistema.

    A posição é composta de:
        - side (1 = long, -1 = short, 0 = flat)
        - size (quantidade)
        - avg_price (preço médio)
        - realized_pnl (lucro já fechado)
        - unrealized_pnl (lucro em aberto)
        - mfe / mae por trade
        - tracking interno para risco e estatísticas

    Regras:
        - repositioning recalcula avg_price corretamente
        - reductions realizam realized_pnl parcial
        - flip (ex: long → short) liquida a antiga e abre outra no mesmo tick
    """

    side: int = 0             # 1 long, -1 short, 0 flat
    size: float = 0.0
    avg_price: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Maximum Favorable / Adverse Excursion do trade atual
    mfe: float = 0.0
    mae: float = 0.0

    # tracking interno
    entry_price: Optional[float] = None

    def is_flat(self) -> bool:
        return self.size == 0

    # ============================================================
    # 1) Abrir nova posição
    # ============================================================
    def open(self, side: int, size: float, price: float):
        """
        Abre posição nova (se estava flat).
        """
        self.side = side
        self.size = size
        self.avg_price = price

        self.entry_price = price
        self.mfe = 0.0
        self.mae = 0.0
        self.unrealized_pnl = 0.0

    # ============================================================
    # 2) Adicionar posição (pyramiding)
    # ============================================================
    def increase(self, size: float, price: float):
        """
        Aumenta posição mantendo preço médio correto.
        """
        new_size = self.size + size
        self.avg_price = (self.avg_price * self.size + price * size) / new_size
        self.size = new_size

    # ============================================================
    # 3) Reduzir posição (partial close)
    # ============================================================
    def reduce(self, size: float, price: float):
        """
        Fecha parte da posição e regista realized PnL.
        """
        if size > self.size:
            raise ValueError("Position.reduce: size superior à posição.")

        pnl_per_unit = (price - self.avg_price) * self.side
        closed_pnl = pnl_per_unit * size

        self.realized_pnl += closed_pnl
        self.size -= size

        if self.size == 0:
            self.side = 0
            self.avg_price = 0.0
            self.entry_price = None
            self.mfe = 0.0
            self.mae = 0.0

        return closed_pnl

    # ============================================================
    # 4) Flip de posição (long → short ou short → long)
    # ============================================================
    def flip(self, new_side: int, size: float, price: float):
        """
        Fecha totalmente a antiga posição e abre nova no mesmo instante.
        """
        if not self.is_flat():
            self.close_all(price)

        self.open(new_side, size, price)

    # ============================================================
    # 5) Fechar posição completamente
    # ============================================================
    def close_all(self, price: float):
        if self.is_flat():
            return 0.0

        pnl_per_unit = (price - self.avg_price) * self.side
        closed_pnl = pnl_per_unit * self.size

        self.realized_pnl += closed_pnl

        # reset
        self.side = 0
        self.size = 0.0
        self.avg_price = 0.0
        self.unrealized_pnl = 0.0
        self.entry_price = None
        self.mfe = 0.0
        self.mae = 0.0

        return closed_pnl

    # ============================================================
    # 6) Atualizar MFE / MAE e Unrealized PnL
    # ============================================================
    def update_unrealized(self, price: float):
        """
        Atualiza lucro não realizado + MFE/MAE.
        """
        if self.is_flat():
            self.unrealized_pnl = 0.0
            return

        pnl = (price - self.avg_price) * self.side
        self.unrealized_pnl = pnl

        # MFE e MAE são sempre relativos à entrada
        excursion = (price - self.entry_price) * self.side

        self.mfe = max(self.mfe, excursion)
        self.mae = min(self.mae, excursion)
