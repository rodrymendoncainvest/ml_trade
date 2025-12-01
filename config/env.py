"""
env.py — Gestão de variáveis de ambiente do backend ML_Trade.

Objetivos:
- carregar API keys externas (Alphavantage, Polygon, Binance, etc.)
- permitir overrides de settings via ambiente
- fornecer flags de execução (ex: DEBUG)
"""

import os
from pathlib import Path

class EnvConfig:
    """
    Carrega variáveis de ambiente do sistema.
    Usado para:
    - providers externos
    - API keys
    - endpoints
    - flags de debug
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # API KEYS (adiciona mais quando necessário)
        # ------------------------------------------------------------------
        self.ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", None)
        self.POLYGON_KEY = os.getenv("POLYGON_KEY", None)
        self.BINANCE_KEY = os.getenv("BINANCE_API_KEY", None)
        self.BINANCE_SECRET = os.getenv("BINANCE_API_SECRET", None)

        # ------------------------------------------------------------------
        # FLAGS DO SISTEMA
        # ------------------------------------------------------------------
        self.DEBUG = bool(int(os.getenv("ML_TRADE_DEBUG", "0")))
        self.USE_GPU = bool(int(os.getenv("ML_TRADE_USE_GPU", "1")))

        # ------------------------------------------------------------------
        # OVERRIDES DOS SETTINGS (opcional)
        # exemplo:
        # ML_TRADE_WINDOW_SIZE=128
        # ------------------------------------------------------------------
        self.OVERRIDE_WINDOW_SIZE = self._load_int("ML_TRADE_WINDOW_SIZE")
        self.OVERRIDE_HORIZON = self._load_int("ML_TRADE_HORIZON")
        self.OVERRIDE_BATCH_SIZE = self._load_int("ML_TRADE_BATCH_SIZE")

        # ------------------------------------------------------------------
        # PATHS externos (caso haja setups especiais)
        # ------------------------------------------------------------------
        self.DATA_ROOT = os.getenv("ML_TRADE_DATA_ROOT", None)

    # ======================================================================
    # MÉTODOS AUXILIARES
    # ======================================================================

    @staticmethod
    def _load_int(var: str):
        """
        Lê um inteiro de uma variável de ambiente.
        Retorna None se não existir.
        """
        value = os.getenv(var, None)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None


# Instância global acessível em todo o backend
ENV = EnvConfig()
