from __future__ import annotations
from typing import Optional, List
from abc import ABC, abstractmethod

from .order import Order
from .position import Position


class Strategy(ABC):
    """
    Classe base para todas as estratégias.

    Cada estratégia recebe:
        - engine (para enviar ordens e aceder à posição)
        - index i (a barra atual)
    
    E tem de devolver:
        - None: sem ação
        - int: sinal (1=long, -1=short, 0=flat)
        - Order: ordem individual
        - list[Order]: várias ordens na mesma vela

    Estrutura industrial:
    ----------------------
    ✔ Base unificada para qualquer tipo de estratégia
    ✔ Método setup() para pré-cálculos (opcional)
    ✔ Método before_bar() para atualizar estado (opcional)
    ✔ Método generate() → obrigatório, produz sinal ou ordens
    ✔ Controlo interno de proteção / SL / TP por estratégia
    ✔ Compatível com Engine V1 e Exchange V1
    """

    def __init__(self, engine, df):
        """
        engine → BacktestEngine
        df     → referência ao DataFrame completo do backtest
        """
        self.engine = engine
        self.df = df
        self.position: Position = engine.exchange.position
        self.i = None  # índice atual da barra

        # para estratégias que queiram buffers internos
        self.state = {}

        self.setup()

    # ================================================================
    #  MÉTODO OPCIONAL — corre uma vez antes do backtest
    # ================================================================
    def setup(self):
        """Override opcional. Ideal para pré-cálculos (indicadores, buffers)."""
        pass

    # ================================================================
    #  MÉTODO OPCIONAL — corre antes da geração de sinal na barra i
    # ================================================================
    def before_bar(self):
        """Override opcional. Corre antes do generate() a cada candle."""
        pass

    # ================================================================
    #  MÉTODO OBRIGATÓRIO — gera sinal ou ordens
    # ================================================================
    @abstractmethod
    def generate(self) -> Optional[object]:
        """
        Estratégias devem devolver UMA das seguintes opções:
        
        - None → nada a fazer
        - int  → sinal discreto (+1, -1, 0)
        - Order → ordem individual
        - list[Order] → várias ordens
        
        A Engine trata automaticamente a forma devolvida.
        """
        pass

    # ================================================================
    #  MÉTODO CHAMADO PELA ENGINE
    # ================================================================
    def __call__(self, engine, i: int):
        """
        Pipeline interno da estratégia:
            1. guardar índice i
            2. correr before_bar()
            3. correr generate()
        """

        self.i = i
        self.before_bar()
        return self.generate()

    # ================================================================
    #  HELPERS PARA ESTRATÉGIAS
    # ================================================================

    def long(self, size: float = 1.0) -> Order:
        """Abre posição long imediatamente."""
        return Order(type="market", side=1, size=size)

    def short(self, size: float = 1.0) -> Order:
        """Abre posição short imediatamente."""
        return Order(type="market", side=-1, size=size)

    def flat(self) -> list:
        """Fecha posição atual (qualquer lado)."""
        pos = self.position
        if pos.is_flat():
            return []
        return [Order(type="market", side=-pos.side, size=pos.size)]

    def reverse(self, size: Optional[float] = None) -> list:
        """
        Inverte posição:
            long → short
            short → long
        """
        pos = self.position

        if pos.is_flat():
            raise ValueError("Strategy.reverse(): não há posição aberta.")
        
        size = size or pos.size
        return [
            Order(type="market", side=-pos.side, size=pos.size * 2)
        ]

    def close_if_sl_hit(self, price: float) -> Optional[Order]:
        """
        Proteção manual de SL.
        A Position já tem SL interno, mas a estratégia pode impor um extra.
        """
        pos = self.position
        if pos.is_flat():
            return None

        if pos.side == 1 and price <= pos.stop_loss:
            return Order(type="market", side=-1, size=pos.size)

        if pos.side == -1 and price >= pos.stop_loss:
            return Order(type="market", side=1, size=pos.size)

        return None

    def close_if_tp_hit(self, price: float) -> Optional[Order]:
        """
        Proteção manual de TP.
        """
        pos = self.position
        if pos.is_flat():
            return None

        if pos.side == 1 and price >= pos.take_profit:
            return Order(type="market", side=-1, size=pos.size)

        if pos.side == -1 and price <= pos.take_profit:
            return Order(type="market", side=1, size=pos.size)

        return None
