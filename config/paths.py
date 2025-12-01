# config/paths.py
import os
from pathlib import Path

"""
paths.py — Infraestrutura oficial e única do ML_Trade

ESTRUTURA FINAL DO PROJETO (fixa e industrial):

storage/
    assets/
        <TICKER>/
            dataset/
                X.npy
                y.npy
                features.parquet
                metadata.json
            models/
                best_model.pth
                train_metadata.json
            scalers/
                scaler.pkl
            backtest/
                equity.png
                drawdown.png
                metrics.txt
            inference/
                predictions.json
                debug/
            logs/
                training.log
                inference.log

Este ficheiro:
✔ Define ROOT absoluto do projeto
✔ Cria automaticamente a estrutura por ticker
✔ Fornece funções para aceder a todos os caminhos relevantes
✔ Substitui completamente AssetPaths (AGORA REMOVIDO)
✔ É a fonte única de paths para *todos* os pipelines
"""

# ======================================================================
# ROOT DO PROJETO (pasta ML_trade/)
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# STORAGE ROOT
# ======================================================================
STORAGE_ROOT = PROJECT_ROOT / "storage"
ASSETS_ROOT = STORAGE_ROOT / "assets"

# garantir existência das pastas base
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
ASSETS_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# FUNÇÃO PRINCIPAL: devolve caminhos completos para 1 TICKER
# ======================================================================
def get_paths(ticker: str) -> dict:
    """
    Retorna todos os caminhos organizados para um ticker específico.
    Cria automaticamente a estrutura industrial caso não exista.
    """

    ticker = ticker.upper()
    base = ASSETS_ROOT / ticker

    paths = {
        "root": base,

        # DATASET (features, X.npy, y.npy, metadata.json)
        "dataset": base / "dataset",

        # MODELOS TREINADOS (best_model, checkpoints, metadata)
        "models": base / "models",

        # SCALERS (normalizer)
        "scalers": base / "scalers",

        # BACKTEST OUTPUTS
        "backtest": base / "backtest",

        # INFERENCE OUTPUTS
        "inference": base / "inference",

        # LOGS
        "logs": base / "logs",
    }

    # Criar todas as pastas
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


# ======================================================================
# CAMINHOS GLOBAIS (RAW DATA / TICKERS CSV)
# ======================================================================
PATH_RAW = PROJECT_ROOT / "data" / "history"
PATH_TICKERS = PROJECT_ROOT / "Tickers_1H"

PATH_RAW.mkdir(parents=True, exist_ok=True)
PATH_TICKERS.mkdir(parents=True, exist_ok=True)
