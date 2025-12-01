# =====================================================================
# pipeline_training.py — Industrial Auto-Model Trainer (GPU Optimized)
# Modelos: TCN → Transformer → NBeats → LSTM → GRU
# Seleção automática pelo menor val_loss
# =====================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

from config.paths import get_paths
from models.networks.registry import ModelRegistry


# ================================================================
# HYPERPARAMETERS (globais)
# ================================================================
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 64
TEST_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ordem de teste automática
MODEL_ORDER = ["TCN", "TRANSFORMER", "NBEATS", "LSTM", "GRU"]


# ================================================================
# LOAD DATASET
# ================================================================
def load_dataset(ticker: str):
    paths = get_paths(ticker)

    X = np.load(paths["dataset"] / "X.npy")
    y = np.load(paths["dataset"] / "y.npy")

    return X.astype(np.float32), y.astype(np.float32)


# ================================================================
# TREINO DE UM ÚNICO MODELO
# ================================================================
def train_single_model(model_name: str, X_train, y_train, X_val, y_val, n_features, horizon, paths):
    print(f"\n🔥 A treinar modelo: {model_name}  | DEVICE = {DEVICE}")

    model = ModelRegistry.create(model_name, input_dim=n_features, horizon=horizon).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # tensores
    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        opt.step()

        # validação
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        print(f"[{model_name}] Epoch {epoch}/{EPOCHS}  | train={loss.item():.6f} | val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), paths["models"] / f"{model_name}_best.pth")

    return best_loss


# ================================================================
# TREINO AUTOMÁTICO (TODOS OS MODELOS)
# ================================================================
def train_all_models(ticker: str):
    print("=======================================================")
    print(f"🔥 TREINO AUTOMÁTICO — {ticker}")
    print("=======================================================\n")

    paths = get_paths(ticker)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    X, y = load_dataset(ticker)

    n_features = X.shape[2]
    horizon = y.shape[1] if len(y.shape) == 2 else 1

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    # --------------------------------------------------------
    # TREINO AUTO-MODEL SELECTOR
    # --------------------------------------------------------
    val_results = {}

    for model_name in MODEL_ORDER:
        print("\n-------------------------------------------------------")
        print(f"➡️  A treinar modelo: {model_name}")
        print("-------------------------------------------------------")

        val_loss = train_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_features=n_features,
            horizon=horizon,
            paths=paths
        )

        val_results[model_name] = val_loss
        print(f"✔ Modelo {model_name} terminado com val_loss = {val_loss:.6f}")

    # --------------------------------------------------------
    # ESCOLHER MELHOR MODELO
    # --------------------------------------------------------
    best_model = min(val_results, key=val_results.get)
    best_loss = val_results[best_model]

    print("\n=======================================================")
    print(f"🏆 MELHOR MODELO PARA {ticker}: {best_model}")
    print(f"📉 val_loss = {best_loss:.6f}")
    print("=======================================================\n")

    # guardar ficheiros
    meta = {
        "ticker": ticker,
        "best_model": best_model,
        "best_val_loss": best_loss,
        "horizon": horizon,
        "n_features": n_features,
        "models_tested": MODEL_ORDER,
        "val_results": val_results,
    }

    Path(paths["models"] / "train_metadata.json").write_text(json.dumps(meta, indent=4))
    Path(paths["logs"] / "val_losses.json").write_text(json.dumps(val_results, indent=4))

    # renomear o best_model.pth final
    final_path = paths["models"] / "best_model.pth"
    best_model_src = paths["models"] / f"{best_model}_best.pth"

    if best_model_src.exists():
        best_model_src.replace(final_path)

    print(f"💾 Modelo final guardado em: {final_path}")

    return best_model, best_loss


# ================================================================
# RUN DIRECTLY
# ================================================================
if __name__ == "__main__":
    train_all_models("GALP.LS")
