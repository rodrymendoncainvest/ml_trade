# ======================================================================
#  BACKTEST PIPELINE V1 — Industrial, limpo e totalmente compatível
# ======================================================================

import numpy as np
import pandas as pd
from pathlib import Path

# --- Data pipeline ---
from data.raw_ingestor import RawIngestor
from data.translator import Translator
from data.aligner import Aligner
from data.validator import Validator
from data.normalizer import Normalizer
from data.features import FeatureGenerator
from data.window import WindowBuilder

# --- Backtest ecosystem ---
from backtest.backtest_engine import BacktestEngine
from backtest.backtest_metrics import BacktestMetrics
from backtest.backtest_reports import BacktestReport
from backtest.strategies import HybridMLStrategy


# --- Config paths ---
from config.paths import get_paths, PATH_RAW


# ======================================================================
#  RUN BACKTEST PIPELINE
# ======================================================================

def run_backtest(
    ticker: str,
    frequency: str = "1h",
    strategy_mode: str = "hybrid",    # "close" | "prediction" | "hybrid"
):
    print("-----------------------------------------------------")
    print(f"BACKTEST PIPELINE V1 — {ticker}")
    print("-----------------------------------------------------\n")

    paths = get_paths(ticker)

    # ==================================================================
    # 1) LOAD RAW
    # ==================================================================
    print("[1] RAW INGESTOR")
    raw_file = PATH_RAW / f"{ticker}_1H.csv"

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Ficheiro RAW não encontrado:\n{raw_file}\n\n"
            "Faz download prévio dos dados para data/history/"
        )

    df = RawIngestor().load_csv(raw_file)
    print(f"→ raw: {len(df)} linhas")

    # ==================================================================
    # 2) TRANSLATE
    # ==================================================================
    print("\n[2] TRANSLATOR")
    df = Translator().translate(df)
    print(f"→ translator: {len(df)} linhas")

    # ==================================================================
    # 3) ALIGN
    # ==================================================================
    print("\n[3] ALIGNER")
    df = Aligner(frequency=frequency).align(df, ticker=ticker)
    print(f"→ align: {len(df)} linhas (synthetic={df['synthetic'].sum()})")

    # ==================================================================
    # 4) VALIDATOR
    # ==================================================================
    print("\n[4] VALIDATOR")
    df = Validator().validate(df)
    print("→ validator OK")

    # ==================================================================
    # 5) NORMALIZER
    # ==================================================================
    print("\n[5] NORMALIZER")
    normalizer = Normalizer(scaler_type="standard", use_logreturn=True, ticker=ticker)
    df_norm = normalizer.normalize(df)
    print(f"→ normalizer: {len(df_norm)} linhas")

    # ==================================================================
    # 6) FEATURES
    # ==================================================================
    print("\n[6] FEATURES")
    df_feat = FeatureGenerator().generate(df_norm)
    print(f"→ features: {len(df_feat)} linhas | {len(df_feat.columns)} cols")

    # ==================================================================
    # 7) STRATEGY SELECTION (industrial)
    # ==================================================================
    print("\n[7] STRATEGY")

    engine = BacktestEngine()

    if strategy_mode == "close":
        signals = engine.build_signals_from_close(df_feat)

    elif strategy_mode == "prediction":
        if "prediction" not in df_feat.columns:
            raise ValueError(
                "Backtest (prediction mode): falta coluna 'prediction'. "
                "Executa primeiro pipeline_inference.py"
            )
        signals = engine.build_signals_from_prediction(df_feat)

    elif strategy_mode == "hybrid":
        signals = HybridMLStrategy().generate(df_feat)

    else:
        raise ValueError("strategy_mode deve ser: close | prediction | hybrid")

    print("→ strategy OK")

    # ==================================================================
    # 8) RUN BACKTEST
    # ==================================================================
    print("\n[8] BACKTEST ENGINE")
    results = engine.run(df_feat, signals)
    print("→ backtest executado")

    # ==================================================================
    # 9) METRICS
    # ==================================================================
    print("\n[9] METRICS")
    metrics = BacktestMetrics.compute(results)
    print("→ métricas calculadas")

    # ==================================================================
    # 10) REPORTS
    # ==================================================================
    print("\n[10] REPORTS")

    reports = BacktestReport(paths["backtest"], ticker)
    reports.generate_full_report(results, metrics)

    print(f"→ relatórios gravados em:\n{paths['backtest']}")

    print("\nBACKTEST PIPELINE — COMPLETO")
    print("-----------------------------------------------------\n")

    return results, metrics


# ======================================================================
# DIRECT EXECUTION
# ======================================================================
if __name__ == "__main__":
    run_backtest(
        ticker="GALP.LS",
        frequency="1h",
        strategy_mode="hybrid",
    )
