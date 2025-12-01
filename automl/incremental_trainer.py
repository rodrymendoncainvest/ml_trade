# ======================================================================
# incremental_trainer.py — Treino incremental AutoML
# Cria versões, continua treino de versões anteriores,
# regista métricas e escolhe melhor modelo automaticamente.
# ======================================================================

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from models.networks.registry import ModelRegistry
from automl.model_tracker import ModelTracker
from config.paths import get_paths


class IncrementalTrainer:
    """
    Treinador AutoML Evolutivo:
    - Cria nova versão automaticamente
    - Usa modelo anterior como base (quando existe)
    - Guarda métricas completas por versão
    - Sincroniza com ModelTracker
    """

    def __init__(
        self,
        ticker: str,
        model_name: str = "TCN",
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = None,
    ):
        self.ticker = ticker.upper()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.paths = get_paths(self.ticker)
        self.tracker = ModelTracker(self.paths["models"])

    # ================================================================
    # LOAD DATASET
    # ================================================================
    def load_dataset(self):
        X = np.load(self.paths["dataset"] / f"{self.ticker}_X.npy")
        y = np.load(self.paths["dataset"] / f"{self.ticker}_y.npy")

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        return X, y

    # ================================================================
    # TREINO DE UMA VERSÃO
    # ================================================================
    def train_version(self):
        print("\n===================================================")
        print(f"🔧 AutoML Incremental — {self.ticker}")
        print("===================================================\n")

        # ------------------------------------------------------------
        # 1) nova versão
        # ------------------------------------------------------------
        version = self.tracker.register_new_version()
        print(f"➡ Criada versão: v{version}")

        # ------------------------------------------------------------
        # 2) Load dataset
        # ------------------------------------------------------------
        X, y = self.load_dataset()
        n_features = X.shape[2]
        horizon = y.shape[1]

        # split simples (não aleatório)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).to(self.device)
        X_val_t   = torch.tensor(X_val).to(self.device)
        y_val_t   = torch.tensor(y_val).to(self.device)

        # ------------------------------------------------------------
        # 3) Build model (ou carregar modelo anterior)
        # ------------------------------------------------------------
        model = ModelRegistry.create(
            self.model_name,
            input_dim=n_features,
            horizon=horizon
        ).to(self.device)

        prev_path = self.paths["models"] / "best_model.pth"

        if prev_path.exists():
            print(f"↪ Modelo anterior encontrado. Continuar treino: {prev_path}")
            model.load_state_dict(torch.load(prev_path, map_location=self.device))
        else:
            print("↪ Nenhum modelo anterior. Treino inicial de raiz.")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # ------------------------------------------------------------
        # 4) Train loop completo
        # ------------------------------------------------------------
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()

            # Validação
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_loss = loss_fn(pred_val, y_val_t).item()

            print(f"Epoch {epoch}/{self.epochs} | loss={loss.item():.6f} | val_loss={val_loss:.6f}")

            # rastrear melhor modelo desta *versão*
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                version_path = self.paths["models"] / f"v{version}"
                version_path.mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), version_path / "model.pth")
                print(f"💾 Guardado checkpoint de versão em {version_path}/model.pth")

        # ------------------------------------------------------------
        # 5) Guardar métricas desta versão
        # ------------------------------------------------------------
        self.tracker.record_training_metrics(version, loss.item(), best_val_loss)

        # hiperparâmetros
        self.tracker.record_hyperparams(version, {
            "model": self.model_name,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size
        })

        # ------------------------------------------------------------
        # 6) Atualizar modelo global "best_model.pth"
        # ------------------------------------------------------------
        winner = self.tracker.update_best_model()
        print("\n🏆 Melhor versão até agora:")
        print(winner)

        # copiar ficheiro vencedor
        best_version = winner["version"]
        best_file = self.paths["models"] / f"v{best_version}" / "model.pth"

        final_path = self.paths["models"] / "best_model.pth"
        torch.save(torch.load(best_file, map_location=self.device), final_path)

        print(f"\n🎯 best_model.pth atualizado → versão v{best_version}")

        print("\n===================================================")
        print("AUTO ML — Treino incremental concluído")
        print("===================================================\n")

        return version, best_version
