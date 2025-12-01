import torch

"""
settings.py — Configurações globais do backend ML_Trade

Este ficheiro define:
- parâmetros do pipeline
- defaults de treino
- configs do sistema
- device (CPU/GPU)
- parâmetros mínimos de dados
"""

SETTINGS = {
    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------
    # janela temporal padrão usada em features + window builder
    "window_size": 64,

    # horizonte de previsão (quantas velas à frente prever por defeito)
    "horizon": 1,

    # requisitos mínimos para validar uma série
    "min_history_points": 500,

    # provider padrão para data_downloader
    "default_provider": "yahoo",

    # tickers principais do sistema (para scanning / batch pipeline)
    "tickers": [
        "AAPL",
        "MSFT",
        "BTC-USD"
    ],

    # ------------------------------------------------------------------
    # TREINO (DEFAULTS)
    # ------------------------------------------------------------------
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "early_stopping_patience": 8,
    "train_val_split": 0.2,

    # ------------------------------------------------------------------
    # HARDWARE
    # ------------------------------------------------------------------
    # seleção automática de GPU se existir
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ------------------------------------------------------------------
    # LOGGING & DEBUG
    # ------------------------------------------------------------------
    # nível de verbosidade do sistema
    "verbose": True,

    # Ativar/desativar prints de debug internos
    "debug_mode": False,
}
