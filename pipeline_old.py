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
from data.window import WindowBuilder

from models.networks.registry import ModelRegistry

from backtest.backtest_engine import BacktestEngine
from backtest.backtest_metrics import BacktestMetrics
from backtest.backtest_reports import BacktestReport
from backtest.strategies.hybrid_ml import HybridMLStrategy


DEVICE = torch.device(SETTINGS.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
EPOCHS = int(SETTINGS.get("epochs", 50))
BATCH_SIZE = int(SETTINGS.get("batch_size", 32))
LEARNING_RATE = float(SETTINGS.get("learning_rate", 1e-4))
TRAIN_VAL_SPLIT = float(SETTINGS.get("train_val_split", 0.2))
EARLY_STOPPING_PATIENCE = int(SETTINGS.get("early_stopping_patience", 8))


def _progress(step_idx: int, total_steps: int, label: str) -> None:
    pct = int(round((step_idx / total_steps) * 100))
    pct = max(0, min(100, pct))
    filled = pct // 10
    bar = "[" + "#" * filled + "." * (10 - filled) + f"] {pct:3d}%  {label}"
    print(bar)
    sys.stdout.flush()
    time.sleep(0.15)


# =====================================================================
# 1. DOWNLOAD
# =====================================================================

def run_download(ticker: str, period: str = "730d") -> None:
    print("-----------------------------------------------------")
    print(f"[1] DOWNLOAD 1H — {ticker}")
    print("-----------------------------------------------------")

    downloader = DataDownloader(period=period)
    downloader.download_1h(ticker)

    print("✔ Download concluído.\n")


# =====================================================================
# 2. PREPROCESS: RAW → translate → align → validate → normalize → features → windows → dataset
# =====================================================================

def run_preprocess(
    ticker: str,
    window_size: int,
    horizon: int,
    frequency: str = "1h",
) -> Tuple[np.ndarray, np.ndarray]:
    print("-----------------------------------------------------")
    print(f"[2] PREPROCESS — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)

    # 1) RAW
    raw_file = PATH_RAW / f"{ticker}_1H.csv"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"RAW não encontrado em {raw_file}. "
            "Corre primeiro o passo de download."
        )

    ingestor = RawIngestor()
    df = ingestor.load_csv(raw_file)
    print(f"→ RAW: {len(df)} linhas")

    # 2) TRANSLATOR
    df = Translator().translate(df)
    print(f"→ Translator: {len(df)} linhas")

    # 3) ALIGNER
    df = Aligner(frequency=frequency).align(df, ticker=ticker)
    print(f"→ Aligner: {len(df)} linhas (synthetic={int(df.get('synthetic', 0).sum())})")

    # 4) DROPNA
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"→ Dropna: removidas {before - len(df)}, restantes {len(df)}")

    # 5) VALIDATOR
    df = Validator().validate(df)
    print("✔ Validator OK")

    # 6) NORMALIZER
    normalizer = Normalizer(scaler_type="standard", use_logreturn=True, ticker=ticker)
    df_norm = normalizer.normalize(df)
    print(f"→ Normalizer: {len(df_norm)} linhas")

    # 7) FEATURES
    feat_gen = FeatureGenerator()
    df_feat = feat_gen.generate(df_norm)
    print(f"→ Features: {len(df_feat)} linhas | {len(df_feat.columns)} colunas")

    # 8) WINDOW BUILDER
    feature_cols = [c for c in df_feat.columns if c != "timestamp"]
    if "close" not in df_feat.columns:
        raise ValueError("Preprocess: coluna 'close' em falta após feature generation.")

    wb = WindowBuilder(
        window_size=window_size,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col="close",
    )

    X, y = wb.build(df_feat)
    print(f"→ WindowBuilder: X={X.shape}, y={y.shape}")

    # 9) SAVE DATASET + METADATA
    ds_dir = paths["dataset"]
    ds_dir.mkdir(parents=True, exist_ok=True)

    features_out = ds_dir / f"{ticker}_features.parquet"
    X_out = ds_dir / f"{ticker}_X.npy"
    y_out = ds_dir / f"{ticker}_y.npy"
    meta_out = ds_dir / f"{ticker}_metadata.json"

    df_feat_reset = df_feat.reset_index(drop=True)
    df_feat_reset.to_parquet(features_out, index=False)
    np.save(X_out, X.astype(np.float32))
    np.save(y_out, y.astype(np.float32))

    metadata: Dict[str, object] = {
        "ticker": ticker,
        "window_size": int(window_size),
        "horizon": int(horizon),
        "frequency": frequency,
        "rows_final": int(len(df_feat_reset)),
        "n_features": int(len(feature_cols)),
    }
    Path(meta_out).write_text(json.dumps(metadata, indent=4))

    print("✔ Preprocess concluído.")
    print(f"   Features : {features_out}")
    print(f"   X dataset: {X_out}")
    print(f"   y dataset: {y_out}")
    print(f"   Metadata : {meta_out}\n")

    return X, y


# =====================================================================
# 3. TRAINING — Auto-model trainer (TCN, TRANSFORMER, NBEATS, LSTM, GRU)
# =====================================================================

MODEL_ORDER = ["TCN", "TRANSFORMER", "NBEATS", "LSTM", "GRU"]


def _load_dataset_for_training(ticker: str) -> Tuple[np.ndarray, np.ndarray]:
    paths = get_paths(ticker)
    ds_dir = paths["dataset"]

    X_path = ds_dir / f"{ticker}_X.npy"
    y_path = ds_dir / f"{ticker}_y.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {ds_dir}. "
            "Corre primeiro o preprocess."
        )

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y


def _train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    horizon: int,
    paths: Dict[str, Path],
) -> float:
    print(f"\n🔥 A treinar modelo: {model_name}  | device={DEVICE}")

    model = ModelRegistry.create(model_name, input_dim=n_features, horizon=horizon).to(DEVICE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    if y_train_t.ndim == 1:
        y_train_t = y_train_t.unsqueeze(-1)
    if y_val_t.ndim == 1:
        y_val_t = y_val_t.unsqueeze(-1)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        print(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} "
            f"| train={loss.item():.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                paths["models"] / f"{model_name}_best.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[{model_name}] early stopping aos {epoch} epochs.")
                break

    return float(best_loss)


def train_all_models(ticker: str) -> Tuple[str, float]:
    print("=======================================================")
    print(f"🔥 TREINO AUTOMÁTICO — {ticker}")
    print("=======================================================\n")

    paths = get_paths(ticker)
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    X, y = _load_dataset_for_training(ticker)

    n_features = X.shape[2]
    horizon = y.shape[1]

    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"→ n_features={n_features}, horizon={horizon}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TRAIN_VAL_SPLIT,
        shuffle=False,
    )

    val_results: Dict[str, float] = {}

    for model_name in MODEL_ORDER:
        try:
            val_loss = _train_single_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_features=n_features,
                horizon=horizon,
                paths=paths,
            )
            val_results[model_name] = val_loss
            print(f"✔ Modelo {model_name} terminado com val_loss={val_loss:.6f}")
        except Exception as e:
            print(f"❌ Falha ao treinar modelo {model_name}: {e}")

    if not val_results:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    best_model = min(val_results, key=val_results.get)
    best_loss = val_results[best_model]

    print("\n=======================================================")
    print(f"🏆 MELHOR MODELO PARA {ticker}: {best_model}")
    print(f"📉 val_loss = {best_loss:.6f}")
    print("=======================================================\n")

    meta = {
        "ticker": ticker,
        "best_model": best_model,
        "best_val_loss": best_loss,
        "horizon": horizon,
        "n_features": n_features,
        "models_tested": MODEL_ORDER,
        "val_results": val_results,
    }

    meta_path = paths["models"] / "train_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=4))

    final_path = paths["models"] / "best_model.pth"
    best_model_src = paths["models"] / f"{best_model}_best.pth"
    if best_model_src.exists():
        best_model_src.replace(final_path)

    print(f"💾 Modelo final guardado em: {final_path}\n")

    return best_model, best_loss


# =====================================================================
# 4. INFERENCE
# =====================================================================

def _load_best_model(ticker: str, input_dim: int, horizon: int):
    paths = get_paths(ticker)
    model_path = paths["models"] / "best_model.pth"
    metadata_path = paths["models"] / "train_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata de treino não encontrada: {metadata_path}")

    meta = pd.read_json(metadata_path, typ="series")
    model_name = str(meta["best_model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] A carregar {model_name} → {device}")

    model = ModelRegistry.create(model_name, input_dim=input_dim, horizon=horizon).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model, device


def run_inference(ticker: str, window_size: int) -> np.ndarray:
    print("-----------------------------------------------------")
    print(f"[4] INFERÊNCIA — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)

    raw_file = PATH_RAW / f"{ticker}_1H.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"RAW não encontrado: {raw_file}")

    df = RawIngestor().load_csv(raw_file)
    print(f"[RAW] {len(df)} linhas")

    df = Translator().translate(df)
    df = Aligner(frequency="1h").align(df, ticker=ticker)
    df = df.dropna().reset_index(drop=True)
    df = Validator().validate(df)

    normalizer = Normalizer(scaler_type="standard", use_logreturn=True, ticker=ticker)
    df_norm = normalizer.normalize(df)

    df_feat = FeatureGenerator().generate(df_norm)
    print(f"[FEATURES] {len(df_feat)} linhas | {len(df_feat.columns)} cols")

    meta_path = paths["dataset"] / f"{ticker}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata do dataset não encontrada: {meta_path}.\n"
            "Corre o preprocess antes de inferência."
        )

    meta = json.loads(Path(meta_path).read_text())
    input_dim = int(meta["n_features"])
    horizon = int(meta["horizon"])

    feature_cols = [c for c in df_feat.columns if c != "timestamp"]
    wb = WindowBuilder(
        window_size=window_size,
        horizon=horizon,
        feature_cols=feature_cols,
        target_col="close",
    )

    window = wb.build_single(df_feat)
    window = np.expand_dims(window, axis=0)

    model, device = _load_best_model(ticker, input_dim=input_dim, horizon=horizon)

    window_tensor = torch.tensor(window, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_norm = model.predict(window_tensor)

    pred_norm = pred_norm.cpu().numpy()[0]

    mean_close = float(normalizer.scaler.mean_[3])
    std_close = float(
        normalizer.scaler.scale_[3]
    ) if hasattr(normalizer.scaler, "scale_") else float(normalizer.scaler.std_[3])

    pred_real = pred_norm * std_close + mean_close

    inf_dir = paths["inference"]
    inf_dir.mkdir(parents=True, exist_ok=True)
    pred_out = inf_dir / "predictions.json"

    pred_series = pd.Series(pred_real, name="prediction")
    pred_series.to_json(pred_out, orient="values")

    print("Previsões (desnormalizadas):")
    for i, p in enumerate(pred_real, start=1):
        print(f"t+{i}: {p}")

    print(f"✔ Inferência concluída. Guardado em: {pred_out}\n")

    return pred_real


# =====================================================================
# 5. BACKTEST
# =====================================================================

def run_backtest(
    ticker: str,
    frequency: str = "1h",
    strategy_mode: str = "hybrid",
):
    print("-----------------------------------------------------")
    print(f"[5] BACKTEST — {ticker}")
    print("-----------------------------------------------------")

    paths = get_paths(ticker)

    raw_file = PATH_RAW / f"{ticker}_1H.csv"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Ficheiro RAW não encontrado:\n{raw_file}\n"
            "Corre primeiro o download."
        )

    df = RawIngestor().load_csv(raw_file)
    print(f"→ RAW: {len(df)} linhas")

    df = Translator().translate(df)
    df = Aligner(frequency=frequency).align(df, ticker=ticker)
    df = df.dropna().reset_index(drop=True)
    df = Validator().validate(df)

    normalizer = Normalizer(scaler_type="standard", use_logreturn=True, ticker=ticker)
    df_norm = normalizer.normalize(df)

    df_feat = FeatureGenerator().generate(df_norm)
    print(f"→ FEATURES: {len(df_feat)} linhas | {len(df_feat.columns)} cols")

    inf_dir = paths["inference"]
    pred_path = inf_dir / "predictions.json"

    if pred_path.exists():
        preds = pd.read_json(pred_path, typ="series").values
        n_pred = len(preds)
        df_feat = df_feat.copy()
        df_feat["prediction"] = 0.0
        df_feat.loc[df_feat.index[-n_pred:], "prediction"] = preds
        print(f"→ Previsões carregadas ({n_pred}) e alinhadas com df_feat.")
    else:
        print("⚠ Nenhum ficheiro de previsões encontrado. strategy_mode='prediction' ou 'hybrid' pode falhar.")

    engine = BacktestEngine()

    if strategy_mode == "close":
        signals = engine.build_signals_from_close(df_feat)
    elif strategy_mode == "prediction":
        if "prediction" not in df_feat.columns:
            raise ValueError("Backtest (prediction): falta coluna 'prediction'.")
        signals = engine.build_signals_from_prediction(df_feat)
    elif strategy_mode == "hybrid":
        strategy = HybridMLStrategy()
        signals = engine.apply_strategy(df_feat, strategy)
    else:
        raise ValueError("strategy_mode deve ser 'close', 'prediction' ou 'hybrid'.")

    results = engine.run(
        df=df_feat,
        signals=signals,
        stop_loss=0.02,
        take_profit=0.04,
    )

    metrics = BacktestMetrics.compute(results)
    bt_dir = paths["backtest"]
    bt_dir.mkdir(parents=True, exist_ok=True)

    reports = BacktestReport(bt_dir, ticker)
    reports.generate_full_report(results, metrics)

    print(f"✔ Backtest concluído. Relatórios em: {bt_dir}\n")

    return results, metrics


# =====================================================================
# 6. PIPELINE COMPLETA + ENTRYPOINT INTERATIVO
# =====================================================================

def run_full_pipeline(ticker: str) -> None:
    """
    Executa toda a pipeline para um ticker:
    download → preprocess → train → inference → backtest
    com barras de progresso.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        print("Ticker inválido.")
        return

    window_size = int(SETTINGS.get("window_size", 64))
    horizon = int(SETTINGS.get("horizon", 1))

    print("\n====================================================")
    print(f"ML TRADE — PIPELINE COMPLETA PARA {ticker}")
    print("====================================================\n")

    total_steps = 5
    step = 1

    _progress(step, total_steps, "DOWNLOAD")
    run_download(ticker)
    step += 1

    _progress(step, total_steps, "PREPROCESS")
    run_preprocess(ticker, window_size=window_size, horizon=horizon, frequency="1h")
    step += 1

    _progress(step, total_steps, "TRAINING")
    train_all_models(ticker)
    step += 1

    _progress(step, total_steps, "INFERENCE")
    run_inference(ticker, window_size=window_size)
    step += 1

    _progress(step, total_steps, "BACKTEST")
    run_backtest(ticker, frequency="1h", strategy_mode="hybrid")

    print("\nPipeline completa terminada para", ticker)
    print("====================================================\n")


if __name__ == "__main__":
    print("\n=== ML TRADE PIPELINE ===")
    try:
        ticker = input("Ticker a correr: ")
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.")
        sys.exit(1)

    run_full_pipeline(ticker)
