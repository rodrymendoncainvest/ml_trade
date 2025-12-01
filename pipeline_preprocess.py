# ======================================================================
#  PREPROCESS PIPELINE — VERSÃO FINAL ROBUSTA
#  Compatível com a tua estrutura real:
#    storage/datasets/
#    storage/features/
#    storage/scalers/
# ======================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# --- Data pipeline modules ---
from data.raw_ingestor import RawIngestor
from data.translator import Translator
from data.aligner import Aligner
from data.validator import Validator
from data.normalizer import Normalizer
from data.features import FeatureGenerator
from data.window import WindowBuilder

# --- Path manager unificado ---
from config.paths import PATH_RAW, get_paths


# ======================================================================
#                     PREPROCESS PIPELINE — MAIN
# ======================================================================
def run_preprocess(
    ticker: str,
    window_size: int = 60,
    horizon: int = 3,
    frequency: str = "1h",
):
    print("-----------------------------------------------------")
    print(f"PREPROCESS PIPELINE — {ticker}")
    print("-----------------------------------------------------\n")

    paths = get_paths(ticker)

    # ==================================================================
    # 1) RAW INGESTOR
    # ==================================================================
    print("[1] RAW INGESTOR")

    raw_file = PATH_RAW / f"{ticker}_1H.csv"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"[ERRO] Ficheiro não encontrado: {raw_file}\n"
            "Corre primeiro: python pipeline_download.py --ticker TICKER"
        )

    df = RawIngestor().load_csv(raw_file)
    print(f"→ raw: {len(df)} linhas")

    # ==================================================================
    # 2) TRANSLATOR
    # ==================================================================
    print("\n[2] TRANSLATOR")
    df = Translator().translate(df)
    print(f"→ translator: {len(df)} linhas")

    # ==================================================================
    # 3) ALIGNER
    # ==================================================================
    print("\n[3] ALIGNER")
    df = Aligner(frequency=frequency).align(df, ticker=ticker)
    print(f"→ alinhado: {len(df)} linhas")
    print(f"→ velas sintéticas: {df['synthetic'].sum()}")

    # ==================================================================
    # 4) DROPNA
    # ==================================================================
    print("\n[4] DROPNA")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"→ removidas: {before - len(df)}")
    print(f"→ restantes: {len(df)}")

    # ==================================================================
    # 5) VALIDATOR
    # ==================================================================
    print("\n[5] VALIDATOR")
    df = Validator().validate(df)
    print("→ validator OK")

    # ==================================================================
    # 6) NORMALIZER
    # ==================================================================
    print("\n[6] NORMALIZER")

    normalizer = Normalizer(
        scaler_type="standard",
        use_logreturn=True,
        ticker=ticker,
    )

    df = normalizer.normalize(df)
    print(f"→ normalizado: {len(df)} linhas")

    # ==================================================================
    # 7) FEATURE GENERATOR
    # ==================================================================
    print("\n[7] FEATURE GENERATION")
    df = FeatureGenerator().generate(df)
    print(f"→ features: {len(df)} linhas")
    print(f"→ nº features: {len(df.columns)}")

    # ==================================================================
    # 8) WINDOW BUILDER (X, y)
    # ==================================================================
    print("\n[8] WINDOW BUILDER")

    feature_cols = [c for c in df.columns if c != "timestamp"]
    wb = WindowBuilder(
        window_size=window_size,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col="close"
    )

    X, y = wb.build(df)
    print(f"→ X shape: {X.shape}")
    print(f"→ y shape: {y.shape}")

    # ==================================================================
    # 9) SAVE ARTIFACTS — NO TEU FORMATO REAL
    # ==================================================================
    print("\n[9] SAVE ARTIFACTS")

    # FEATURES
    features_out = paths["dataset"] / f"{ticker}_features.parquet"
    df.to_parquet(features_out, index=False)

    # DATASET X/Y
    X_out = paths["dataset"] / f"{ticker}_X.npy"
    y_out = paths["dataset"] / f"{ticker}_y.npy"
    np.save(X_out, X)
    np.save(y_out, y)

    # METADATA
    md_out = paths["dataset"] / f"{ticker}_metadata.json"
    metadata = {
        "ticker": ticker,
        "window_size": window_size,
        "horizon": horizon,
        "frequency": frequency,
        "rows_final": len(df),
        "n_features": len(feature_cols),
    }
    pd.Series(metadata).to_json(md_out)

    print(f"→ Features:   {features_out}")
    print(f"→ X dataset:  {X_out}")
    print(f"→ y dataset:  {y_out}")
    print(f"→ Metadata:   {md_out}")

    print("\nPREPROCESS PIPELINE — COMPLETO")
    print("-----------------------------------------------------\n")


# ======================================================================
# EXECUÇÃO
# ======================================================================
if __name__ == "__main__":
    run_preprocess(
        ticker="GALP.LS",
        window_size=60,
        horizon=3,
        frequency="1h",
    )
