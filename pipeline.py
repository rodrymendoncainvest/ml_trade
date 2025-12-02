import sys
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config.settings import SETTINGS
from config.paths import get_paths, PATH_RAW

from data.data_downloader import DataDownloader
from data.raw_ingestor import RawIngestor
from data.translator import Translator
from data.aligner import Aligner
from data.validator import Validator
from data.normalizer import Normalizer
from data.features import FeatureGenerator
from data.feature_scaler import FeatureScaler
from data.window import WindowBuilder
from data.denormalizer import PriceDenormalizer

from models.networks.registry import ModelRegistry

from backtest.backtest_engine import BacktestEngine
from backtest.backtest_metrics import BacktestMetrics
from backtest.backtest_reports import BacktestReport
from backtest.strategies.hybrid_ml import HybridMLStrategy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = int(SETTINGS.get("epochs", 50))
LEARNING_RATE = float(SETTINGS.get("learning_rate", 1e-4))
TRAIN_VAL_SPLIT = float(SETTINGS.get("train_val_split", 0.2))
EARLY_STOPPING_PATIENCE = int(SETTINGS.get("early_stopping_patience", 8))


# ================================================================
def _progress(step_idx: int, total_steps: int, label: str) -> None:
    pct = int(round((step_idx / total_steps) * 100))
    filled = pct // 10
    bar = "[" + "#" * filled + "." * (10 - filled) + f"] {pct:3d}%  {label}"
    print(bar)
    sys.stdout.flush()
    time.sleep(0.15)
# =====================================================================
# 1. DOWNLOAD
# =====================================================================

def run_download(ticker: str, period="730d"):
    print("-----------------------------------------------------")
    print(f"[1] DOWNLOAD 1H — {ticker}")
    print("-----------------------------------------------------")
    DataDownloader(period=period).download_1h(ticker)
    print("✔ Download concluído.\n")


# =====================================================================
# 2. PREPROCESS
# =====================================================================

def run_preprocess(ticker: str, window_size: int, horizon: int, frequency="1h"):
    print("-----------------------------------------------------")
    print(f"[2] PREPROCESS — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)
    raw = PATH_RAW / f"{ticker}_1H.csv"
    df = RawIngestor().load_csv(raw)
    df = Translator().translate(df)
    df = Aligner(frequency=frequency).align(df, ticker=ticker)
    df = df.dropna().reset_index(drop=True)
    df = Validator().validate(df)

    # Normalizer (agora carrega OU faz fit corretamente)
    normalizer = Normalizer("standard", True, ticker)
    df_norm = normalizer.normalize(df)

    feat = FeatureGenerator().generate(df_norm)

    # Feature Scaler
    scaler_path = Path(paths["scalers"]) / f"{ticker}_features_scaler.pkl"
    fs = FeatureScaler(scaler_path)
    feat_scaled = fs.fit_transform(feat.drop(columns=["timestamp"], errors="ignore"))
    feat_scaled["timestamp"] = feat["timestamp"]

    feature_cols = [c for c in feat_scaled.columns if c != "timestamp"]

    wb = WindowBuilder(window_size, horizon, feature_cols, "close")
    X, y = wb.build(feat_scaled)

    ds = paths["dataset"]
    ds.mkdir(exist_ok=True)

    feat_scaled.to_parquet(ds / f"{ticker}_features.parquet", index=False)
    np.save(ds / f"{ticker}_X.npy", X.astype(np.float32))
    np.save(ds / f"{ticker}_y.npy", y.astype(np.float32))

    meta = {
        "ticker": ticker,
        "window_size": window_size,
        "horizon": horizon,
        "n_features": len(feature_cols)
    }
    Path(ds / f"{ticker}_metadata.json").write_text(json.dumps(meta, indent=4))

    print("✔ Preprocess concluído.\n")
    return X, y


# =====================================================================
# 3. TRAINING
# =====================================================================

MODEL_ORDER = ["TCN", "TRANSFORMER", "NBEATS", "LSTM", "GRU"]


def _load_dataset_for_training(ticker: str):
    paths = get_paths(ticker)
    X = np.load(paths["dataset"] / f"{ticker}_X.npy").astype(np.float32)
    y = np.load(paths["dataset"] / f"{ticker}_y.npy").astype(np.float32)
    return X, y


def _train_single(model_name, X_train, y_train, X_val, y_val, n_features, horizon, window_size, paths):
    print(f"\n🔥 A treinar {model_name}")

    model = ModelRegistry.create(model_name, input_dim=n_features, horizon=horizon, window_size=window_size).to(DEVICE)

    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    best = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val_t), y_val_t).item()

        if v < best:
            best = v
            patience = 0
            torch.save(model.state_dict(), paths["models"] / f"{model_name}_best.pth")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    return best


def train_all_models(ticker: str):
    paths = get_paths(ticker)
    paths["models"].mkdir(exist_ok=True)

    X, y = _load_dataset_for_training(ticker)
    n_features = X.shape[2]
    horizon = y.shape[1]
    window_size = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TRAIN_VAL_SPLIT, shuffle=False)

    results = {}
    for model_name in MODEL_ORDER:
        results[model_name] = _train_single(model_name, X_train, y_train, X_val, y_val, n_features, horizon, window_size, paths)

    best = min(results, key=results.get)
    (paths["models"] / f"{best}_best.pth").replace(paths["models"] / "best_model.pth")

    meta = {
        "best_model": best,
        "n_features": n_features,
        "horizon": horizon,
        "window_size": window_size
    }
    Path(paths["models"] / "train_metadata.json").write_text(json.dumps(meta, indent=4))

    return best, results[best]


# =====================================================================
# 4. INFERENCE
# =====================================================================

def _load_best_model(ticker: str, input_dim, horizon, window_size):
    paths = get_paths(ticker)
    meta = json.loads((paths["models"] / "train_metadata.json").read_text())
    name = meta["best_model"]

    model = ModelRegistry.create(name, input_dim=input_dim, horizon=horizon, window_size=window_size).to(DEVICE)
    model.load_state_dict(torch.load(paths["models"] / "best_model.pth", map_location=DEVICE))
    return model


def run_inference(ticker: str, window_size: int):
    print("-----------------------------------------------------")
    print(f"[4] INFERÊNCIA — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)

    df = RawIngestor().load_csv(PATH_RAW / f"{ticker}_1H.csv")
    df = Translator().translate(df)
    df = Aligner("1h").align(df, ticker=ticker)
    df = Validator().validate(df)

    # Normalizer — agora carrega scaler gravado
    norm = Normalizer("standard", True, ticker)
    df_norm = norm.normalize(df)

    denorm = PriceDenormalizer(norm.scaler)

    feat = FeatureGenerator().generate(df_norm)

    fs = FeatureScaler(Path(paths["scalers"]) / f"{ticker}_features_scaler.pkl")
    feat_scaled = fs.transform(feat.drop(columns=["timestamp"], errors="ignore"))
    feat_scaled["timestamp"] = feat["timestamp"]

    meta = json.loads((paths["dataset"] / f"{ticker}_metadata.json").read_text())

    feature_cols = [c for c in feat_scaled.columns if c != "timestamp"]
    wb = WindowBuilder(window_size, meta["horizon"], feature_cols, "close")

    window = wb.build_single(feat_scaled)
    window = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model = _load_best_model(ticker, meta["n_features"], meta["horizon"], window_size)

    with torch.no_grad():
        pred_norm = model.predict(window)[0].cpu().numpy()

    pred_real = np.array([denorm.inverse(v) for v in pred_norm])

    paths["inference"].mkdir(exist_ok=True)
    Path(paths["inference"] / "predictions.json").write_text(json.dumps(pred_real.tolist()))

    print("✔ Inferência concluída")
    return pred_real


# =====================================================================
# 5. BACKTEST
# =====================================================================

def run_backtest(ticker: str, frequency="1h", strategy_mode="hybrid"):
    print("-----------------------------------------------------")
    print(f"[5] BACKTEST — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)

    df = RawIngestor().load_csv(PATH_RAW / f"{ticker}_1H.csv")
    df = Translator().translate(df)
    df = Aligner(frequency).align(df, ticker=ticker)
    df = Validator().validate(df)

    norm = Normalizer("standard", True, ticker)
    df_norm = norm.normalize(df)

    feat = FeatureGenerator().generate(df_norm)

    fs = FeatureScaler(Path(paths["scalers"]) / f"{ticker}_features_scaler.pkl")
    feat_scaled = fs.transform(feat.drop(columns=["timestamp"], errors="ignore"))
    feat_scaled["timestamp"] = feat["timestamp"]

    pred_file = paths["inference"] / "predictions.json"
    preds = json.loads(pred_file.read_text()) if pred_file.exists() else []

    feat_scaled["prediction"] = 0.0
    if preds:
        feat_scaled.iloc[-len(preds):, feat_scaled.columns.get_loc("prediction")] = preds

    engine = BacktestEngine()

    if strategy_mode == "close":
        signals = engine.build_signals_from_close(feat_scaled)
    elif strategy_mode == "prediction":
        signals = engine.build_signals_from_prediction(feat_scaled)
    else:
        signals = engine.apply_strategy(feat_scaled, HybridMLStrategy())

    results = engine.run(feat_scaled, signals)
    metrics = BacktestMetrics.compute(results)

    BacktestReport(paths["backtest"], ticker).generate_full_report(results, metrics)

    print("✔ Backtest concluído")
    return results, metrics


# =====================================================================
# 6. PIPELINE COMPLETA
# =====================================================================

def run_full_pipeline(ticker: str):
    ticker = ticker.upper().strip()

    w = SETTINGS.get("window_size", 64)
    h = SETTINGS.get("horizon", 1)

    print("\n====================================================")
    print(f"ML TRADE — PIPELINE COMPLETA PARA {ticker}")
    print("====================================================\n")

    _progress(1, 5, "DOWNLOAD")
    run_download(ticker)

    _progress(2, 5, "PREPROCESS")
    run_preprocess(ticker, w, h)

    _progress(3, 5, "TRAINING")
    train_all_models(ticker)

    _progress(4, 5, "INFERENCE")
    run_inference(ticker, w)

    _progress(5, 5, "BACKTEST")
    run_backtest(ticker)

    print("\n✔ Pipeline completa terminada.\n")


if __name__ == "__main__":
    print("\n=== ML TRADE PIPELINE ===")
    ticker = input("Ticker a correr: ")
    run_full_pipeline(ticker)
