# automl/model_selector.py
import json
import numpy as np
import torch
from pathlib import Path

from models import ModelRegistry
from backtest.backtest_engine import BacktestEngine
from backtest.backtest_metrics import BacktestMetrics
from automl.meta_features import MetaFeatures
from config.paths import PATHS


class ModelSelector:
    """
    AutoML industrial — treina todos os modelos registados no ModelRegistry:
        - TCN
        - LSTM
        - GRU
        - Transformer
        - NBEATS

    Passos:
        1. Treina cada modelo
        2. Valida
        3. Backtest
        4. Extrai métricas
        5. Converte para meta-features
        6. Calcula score H4
        7. Elege o melhor modelo
        8. Guarda best_model.pth + best_metrics.json + model_name.txt
    """

    def __init__(self, ticker, epochs=40, lr=1e-3, batch_size=64, device=None):
        self.ticker = ticker
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.X = np.load(PATHS["datasets"] / f"{ticker}_X.npy").astype(np.float32)
        self.Y = np.load(PATHS["datasets"] / f"{ticker}_y.npy").astype(np.float32)

        self.input_dim = self.X.shape[2]
        self.horizon = self.Y.shape[1] if len(self.Y.shape) == 2 else 1

        # Prepara pastas
        self.model_dir = Path(PATHS["models"]) / ticker
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    #  TREINAR UM MODELO
    # ============================================================
    def train_single_model(self, model_name):
        print(f"\n🔥 Training {model_name} for {self.ticker}")

        model = ModelRegistry.create(
            model_name,
            input_dim=self.input_dim,
            horizon=self.horizon
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(self.X).to(self.device)
        Y_t = torch.tensor(self.Y).to(self.device)

        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad()

            pred = model(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            optimizer.step()

            print(f"[{model_name}] Epoch {epoch}/{self.epochs} — Loss = {loss.item():.6f}")

        return model

    # ============================================================
    #  AVALIAR MODELO COM BACKTEST
    # ============================================================
    def evaluate_model(self, model):
        print("→ Running backtest...")

        df = np.load(PATHS["datasets"] / f"{self.ticker}_X.npy")

        # Modelo gera previsões normalizadas usando o último batch
        with torch.no_grad():
            model.eval()
            pred = model(torch.tensor(self.X).to(self.device))
            pred = pred.cpu().numpy().flatten()

        # Mock dataframe de previsão
        df_bt = {
            "close": self.X[:, -1, 3],  # close normalizado
            "prediction": pred
        }
        df_bt = np.array(df_bt)

        # Backtest
        engine = BacktestEngine()
        signals = engine.build_signals_from_prediction({"prediction": pred, "close": df_bt["close"]})
        bt = engine.run({"prediction": pred, "close": df_bt["close"]}, signals)

        return BacktestMetrics.compute(bt)

    # ============================================================
    #  AUTO-ML PRINCIPAL
    # ============================================================
    def run(self):
        results = {}

        for model_name in ModelRegistry.available_models():
            print("\n====================================================")
            print(f"   MODEL: {model_name}")
            print("====================================================")

            # 1) Treinar
            model = self.train_single_model(model_name)

            # 2) Avaliar com backtest
            metrics = self.evaluate_model(model)

            # 3) Converter em meta-features
            meta = MetaFeatures.aggregate(metrics)
            score = meta["h4_score"]

            print(f"→ H4 SCORE = {score:.4f}")

            # 4) Guardar tudo
            torch.save(model.state_dict(), self.model_dir / f"{model_name}.pth")

            results[model_name] = {
                "metrics": metrics,
                "meta": meta,
                "score": score,
            }

        # ======================================================
        #  ELEGER MELHOR MODELO
        # ======================================================
        best = max(results.keys(), key=lambda m: results[m]["score"])
        best_model_path = self.model_dir / f"best_model.pth"

        torch.save(
            torch.load(self.model_dir / f"{best}.pth"),
            best_model_path
        )

        with open(self.model_dir / "best_model_name.txt", "w") as f:
            f.write(best)

        with open(self.model_dir / "best_metrics.json", "w") as f:
            json.dump(results[best], f, indent=4)

        print("\n====================================================")
        print(f"🏆 MELHOR MODELO PARA {self.ticker}: {best}")
        print(f"→ Score H4 = {results[best]['score']:.4f}")
        print("====================================================")

        return best, results
