# ======================================================================
#  pipeline_full.py — Sistema completo de ponta a ponta
#  Download → Preprocess → Treino → Inferência → Backtest → Relatórios
# ======================================================================

import time
import numpy as np
import pandas as pd

from data.data_downloader import DataDownloader
from preprocess.pipeline_preprocess import run_preprocess
from training.pipeline_training import train_model
from inference.pipeline_inference import run_inference
from backtest.pipeline_backtest import run_backtest
from config.paths import get_paths


# ======================================================================
#  EXECUTAR TUDO DE FORMA AUTOMÁTICA
# ======================================================================
def run_full_pipeline(
    ticker: str,
    window_size: int = 60,
    horizon: int = 3,
    frequency: str = "1h",
):
    start_total = time.time()

    print("\n====================================================")
    print(f"🔥 ML TRADE — PIPELINE COMPLETO PARA {ticker}")
    print("====================================================\n")

    # =============================================================
    # 1) DOWNLOAD
    # =============================================================
    print("\n----------------------------------------------------")
    print("[1] DOWNLOAD 1H")
    print("----------------------------------------------------")
    downloader = DataDownloader(period="730d")
    downloader.download_1h(ticker)

    # =============================================================
    # 2) PREPROCESS
    # =============================================================
    print("\n----------------------------------------------------")
    print("[2] PREPROCESS")
    print("----------------------------------------------------")

    run_preprocess(
        ticker=ticker,
        window_size=window_size,
        horizon=horizon,
        frequency=frequency,
    )

    # =============================================================
    # 3) TREINO
    # =============================================================
    print("\n----------------------------------------------------")
    print("[3] TREINO DE MODELO")
    print("----------------------------------------------------")

    train_model(ticker)

    # =============================================================
    # 4) INFERÊNCIA
    # =============================================================
    print("\n----------------------------------------------------")
    print("[4] INFERÊNCIA FINAL")
    print("----------------------------------------------------")

    preds = run_inference(ticker, window_size=window_size)

    # =============================================================
    # 5) BACKTEST
    # =============================================================
    print("\n----------------------------------------------------")
    print("[5] BACKTEST AUTOMÁTICO")
    print("----------------------------------------------------")

    run_backtest(
        ticker=ticker,
        signal_mode="hybrid",  # BEST DEFAULT
        frequency=frequency,
    )

    # =============================================================
    # 6) FINAL
    # =============================================================
    total_time = time.time() - start_total

    print("\n====================================================")
    print("🎯 PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print(f"TICKER:      {ticker}")
    print(f"PREVISÃO:    {preds}")
    print(f"TEMPO TOTAL: {round(total_time, 2)}s")
    print("====================================================\n")

    return preds


# ======================================================================
# EXECUÇÃO DIRETA
# ======================================================================
if __name__ == "__main__":
    run_full_pipeline(
        ticker="GALP.LS",
        window_size=60,
        horizon=3,
        frequency="1h",
    )
