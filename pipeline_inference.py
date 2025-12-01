# =====================================================================
# pipeline_inference.py — Inferência Industrial + GPU + Consistência Total
# Usa exatamente a mesma arquitetura do treino e do preprocess
# =====================================================================

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config.paths import get_paths, PATH_RAW

# --- Data pipeline ---
from data.raw_ingestor import RawIngestor
from data.translator import Translator
from data.aligner import Aligner
from data.validator import Validator
from data.normalizer import Normalizer
from data.features import FeatureGenerator
from data.window import WindowBuilder

# --- Models (toda a registry) ---
from models.networks.registry import ModelRegistry


# ============================================================================
# 1) CARREGAR O MELHOR MODELO TREINADO
# ============================================================================
def load_best_model(ticker: str, input_dim: int, horizon: int):
    paths = get_paths(ticker)
    model_path = paths["models"] / "best_model.pth"
    metadata_path = paths["models"] / "train_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Modelo não encontrado:\n{model_path}")

    meta = pd.read_json(metadata_path, typ="series")
    model_name = meta["best_model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] A carregar {model_name} → {device}")

    model = ModelRegistry.create(model_name, input_dim=input_dim, horizon=horizon).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


# ============================================================================
# 2) PIPELINE COMPLETO DE INFERÊNCIA
# ============================================================================
def run_inference(ticker: str, window_size: int = 60):
    print("-----------------------------------------------------")
    print(f"INFERÊNCIA — {ticker}")
    print("-----------------------------------------------------\n")

    paths = get_paths(ticker)

    # =====================================================
    # RAW
    # =====================================================
    ingestor = RawIngestor()

    raw_file = PATH_RAW / f"{ticker}_1H.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"❌ RAW não encontrado: {raw_file}")

    df = ingestor.load_csv(raw_file)
    print(f"[RAW] {len(df)} linhas")

    # =====================================================
    # TRANSLATE
    # =====================================================
    df = Translator().translate(df)
    print("[TRANSLATE] OK")

    # =====================================================
    # ALIGN
    # =====================================================
    df = Aligner(frequency="1h").align(df, ticker=ticker)
    print(f"[ALIGN] {len(df)} linhas")

    # =====================================================
    # VALIDATOR
    # =====================================================
    df = Validator().validate(df)
    print("[VALIDATOR] OK")

    # =====================================================
    # NORMALIZER (carrega scaler guardado no preprocess)
    # =====================================================
    normalizer = Normalizer(
        scaler_type="standard",
        use_logreturn=True,
        ticker=ticker,
    )
    df_norm = normalizer.normalize(df)
    print(f"[NORMALIZER] {len(df_norm)} linhas")

    # =====================================================
    # FEATURES
    # =====================================================
    df_feat = FeatureGenerator().generate(df_norm)
    print(f"[FEATURES] {len(df_feat)} linhas | {len(df_feat.columns)} features")

    # =====================================================
    # WINDOW BUILDER
    # =====================================================
    feature_cols = [c for c in df_feat.columns if c != "timestamp"]

    wb = WindowBuilder(
        window_size=window_size,
        horizon=1,  # o horizon real virá do metadata
        feature_cols=feature_cols,
        target_col="close",
    )

    window = wb.build_single(df_feat)
    window = np.expand_dims(window, axis=0)
    print(f"[WINDOW] {window.shape}")

    # =====================================================
    # METADATA (definida no preprocess)
    # =====================================================
    meta_path = paths["dataset"] / "metadata.json"
    meta = pd.read_json(meta_path, typ="series")

    input_dim = meta["num_features"]
    horizon = meta["horizon"]

    print(f"[META] input_dim={input_dim} | horizon={horizon}")

    # =====================================================
    # CARREGAR MODELO
    # =====================================================
    model, device = load_best_model(ticker, input_dim=input_dim, horizon=horizon)

    window_tensor = torch.tensor(window, dtype=torch.float32).to(device)

    # =====================================================
    # INFERÊNCIA
    # =====================================================
    with torch.no_grad():
        pred_norm = model.predict(window_tensor)

    pred_norm = pred_norm.cpu().numpy()[0]

    # =====================================================
    # DESNORMALIZAÇÃO DO CLOSE
    # =====================================================
    mean_close = normalizer.scaler.mean_[3]
    std_close = normalizer.scaler.std_[3]

    pred_real = pred_norm * std_close + mean_close

    # =====================================================
    # RESULTADOS
    # =====================================================
    print("\n============== PREVISÃO FINAL ==============")
    print(f"Último preço real: {df['close'].iloc[-1]}")
    print("--------------------------------------------")

    for i, p in enumerate(pred_real):
        print(f"t+{i+1}: {p}")

    print("============================================\n")

    return pred_real


# ============================================================================
# EXECUTAR DIRETAMENTE
# ============================================================================
if __name__ == "__main__":
    run_inference("GALP.LS", window_size=60)
