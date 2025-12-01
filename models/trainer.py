import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================================================
# DATASET PARA TREINO — MULTI-TASK (REGRESSÃO + CLASSIFICAÇÃO)
# =========================================================================================

class SequenceDataset(Dataset):
    """
    Dataset industrial para forecasting:
        X → (window, features)
        y_reg → (horizon)
        y_cls → classe derivada do retorno acumulado até t+horizon
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y, dtype=torch.float32)

        # classificação: 0 = down, 1 = flat, 2 = up
        ret = y[:, -1]  # último passo do horizonte
        self.y_cls = torch.empty(len(ret), dtype=torch.long)

        self.y_cls[ret > 0] = 2     # UP
        self.y_cls[ret < 0] = 0     # DOWN
        self.y_cls[ret == 0] = 1    # FLAT

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


# =========================================================================================
# TREINADOR INDUSTRIAL
# =========================================================================================

class Trainer:
    """
    Trainer industrial para modelos TCN / GRU / Transformer.

    Componentes:
        - multi-loss (regressão + classificação)
        - early stopping
        - checkpoints
        - logging automático
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        batch_size: int = 64,
        alpha: float = 0.7,   # peso regressão
        beta: float = 1.0,    # peso classificação
        patience: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "storage/models",
        ticker: str = "GENERIC"
    ):
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.patience = patience
        self.device = device
        self.save_dir = Path(save_dir)
        self.ticker = ticker

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_reg = nn.MSELoss()
        self.loss_cls = nn.CrossEntropyLoss()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.save_dir / f"{ticker}_best_model.pt"

    # ----------------------------------------------------------------------------
    # TRAIN LOOP
    # ----------------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val, epochs=200):

        train_ds = SequenceDataset(X_train, y_train)
        val_ds = SequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_loss = float("inf")
        patience_counter = 0

        history = {"train_loss": [], "val_loss": []}

        print("\n========== TREINO INICIADO ==========\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0

            for Xb, y_reg_b, y_cls_b in train_loader:
                Xb = Xb.to(self.device)
                y_reg_b = y_reg_b.to(self.device)
                y_cls_b = y_cls_b.to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(Xb)     # → (batch, horizon)

                loss_r = self.loss_reg(pred, y_reg_b)
                loss_c = self.loss_cls(pred[:, -1].unsqueeze(1).repeat(1, 3), y_cls_b)

                loss = self.alpha * loss_r + self.beta * loss_c
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # ---------------- VALIDAÇÃO ----------------
            val_loss = self.evaluate(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("\nEARLY STOPPING acionado.\n")
                    break

        print("\n========== TREINO TERMINADO ==========\n")
        return history

    # ----------------------------------------------------------------------------
    # AVALIAÇÃO
    # ----------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total = 0
        for Xb, y_reg_b, y_cls_b in loader:
            Xb = Xb.to(self.device)
            y_reg_b = y_reg_b.to(self.device)
            y_cls_b = y_cls_b.to(self.device)

            pred = self.model(Xb)

            loss_r = self.loss_reg(pred, y_reg_b)
            loss_c = self.loss_cls(pred[:, -1].unsqueeze(1).repeat(1, 3), y_cls_b)

            loss = self.alpha * loss_r + self.beta * loss_c
            total += loss.item()

        return total / len(loader)

    # ----------------------------------------------------------------------------
    # CARREGAR MODELO
    # ----------------------------------------------------------------------------
    def load_best(self):
        if self.ckpt_path.exists():
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"Modelo carregado: {self.ckpt_path}")
        else:
            print("Nenhum checkpoint encontrado.")


