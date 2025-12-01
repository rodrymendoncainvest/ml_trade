"""
registry.py — Registo global de estratégias

Objetivo:
---------
Um ponto único onde:
    • estratégias são registadas por nome
    • o Engine pode instanciá-las dinamicamente
    • o utilizador pode escolher a estratégia por string ("ml", "trend", etc.)
    • é impossível criar nomes duplicados (proteção industrial)

Funcionamento:
-------------
O utilizador (ou pipeline) chama:

    from backtest.registry import STRATEGY_REGISTRY, get_strategy_class
    cls = get_strategy_class("ml")
    strategy = cls(engine, df)

Assim não precisamos hard-code dentro do Engine.
"""

from __future__ import annotations
from typing import Dict, Type
from .strategy_base import Strategy


# =====================================================================
#  DICIONÁRIO GLOBAL DE ESTRATÉGIAS
# =====================================================================
STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}


# =====================================================================
#  FUNÇÃO DE REGISTO — USADA POR CADA FICHEIRO DE STRATEGY
# =====================================================================
def register_strategy(name: str):
    """
    Decorador para registar uma estratégia.

    Exemplo:

        @register_strategy("ml")
        class MLStrategy(Strategy):
            ...

    • garante que o nome está em lowercase
    • impede registos duplicados
    • adiciona ao STRATEGY_REGISTRY
    """

    def decorator(cls: Type[Strategy]):
        key = name.lower().strip()

        if key in STRATEGY_REGISTRY:
            raise ValueError(
                f"Strategy '{key}' já está registada. "
                f"Conflito entre {STRATEGY_REGISTRY[key].__name__} e {cls.__name__}."
            )

        if not issubclass(cls, Strategy):
            raise TypeError(
                f"Classe '{cls.__name__}' não herda de Strategy."
            )

        STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator


# =====================================================================
#  OBTÉM A CLASSE DE UMA ESTRATÉGIA PELO NOME
# =====================================================================
def get_strategy_class(name: str) -> Type[Strategy]:
    """
    Pesquisa industrial:
    - nome é normalizado para lowercase
    - lança erro claro se não existir
    """
    key = name.lower().strip()

    if key not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Estratégia '{key}' não encontrada. "
            f"Estratégias disponíveis: {list(STRATEGY_REGISTRY.keys())}"
        )

    return STRATEGY_REGISTRY[key]


# =====================================================================
#  ÚTIL PARA DEBUG
# =====================================================================
def list_strategies():
    """Devolve lista de estratégias registadas."""
    return list(STRATEGY_REGISTRY.keys())
