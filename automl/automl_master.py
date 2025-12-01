# ======================================================================
# automl_master.py — Orquestrador AutoML completo
# Corre preprocess + treino incremental + backtest + atualização do best model
# ======================================================================

import os
import time
from pathlib import Path

from config.paths import get_paths, PATH_RAW

# Pipelines individuais
from pipeline_preprocess import run_preprocess
from pipeline_backtest import run_backtest
from pipeline_inference import run_inference

# AutoML incremental
from automl.incremental_trainer import IncrementalTrainer


class AutoMLMaster:
    """
    O cérebro do sistema ML_Trade.
    Executa o ciclo completo:
        - preprocess (se necessário)
        - treino incremental
        - backtest
        - escolha automática do melhor modelo
        - inferência
    Tudo isto com um único comando.
    """

    def __init__(
        self,
        ticker: str,
        window_size: int = 60,
        horizon: int = 3,
        frequency: str = "1h",
        model_name: str = "TCN",
    ):
        self.ticker = ticker.upper()
        self.window_size = window_size
        self.horizon = horizon
        self.frequency = frequency
        self.model_name = model_name

        self.paths = get_paths(self.ticker)

    # ==================================================================
    # 1) PREPROCESS — só corre se faltar dataset
    # ==================================================================
    def ensure_preprocess(self):
        dataset_file = self.paths["dataset"] / f"{self.ticker}_X.npy"

        if dataset_file.exists():
            print("💾 Dataset encontrado — preprocess NÃO necessário.")
            return

        print("⚙️ Dataset não encontrado. A correr preprocess...")
        run_preprocess(
            ticker=self.ticker,
            window_size=self.window_size,
            horizon=self.horizon,
            frequency=self.frequency
        )

    # ==================================================================
    # 2) TREINO INCREMENTAL — núcleo AutoML
    # ==================================================================
    def train(self):
        print("\n===================================================")
        print(f"AUTOML — TREINO INCREMENTAL PARA {self.ticker}")
        print("===================================================\n")

        trainer = IncrementalTrainer(
            ticker=self.ticker,
            model_name=self.model_name,
            epochs=20,
            lr=1e-3,
            batch_size=64
        )

        version, best_version = trainer.train_version()

        print(f"✔ Versão treinada: v{version}")
        print(f"✔ Melhor versão atual: v{best_version}")

    # ==================================================================
    # 3) BACKTEST — corre sempre após treino
    # ==================================================================
    def run_backtest(self):
        print("\n===================================================")
        print("AUTOML — A correr backtest do melhor modelo...")
        print("===================================================\n")

        run_backtest(
            ticker=self.ticker,
            frequency=self.frequency,
            window_size=self.window_size,
            horizon=self.horizon
        )

    # ==================================================================
    # 4) INFERÊNCIA — opcional
    # ==================================================================
    def inference(self):
        print("\n===================================================")
        print("AUTOML — Inferência com best_model.pth")
        print("===================================================\n")

        return run_inference(self.ticker, window_size=self.window_size)

    # ==================================================================
    # 5) EXECUÇÃO TOTAL
    # ==================================================================
    def run_all(self, do_inference=True):
        print("\n===================================================")
        print(f"🚀 AUTO ML MASTER — CICLO COMPLETO ({self.ticker})")
        print("===================================================\n")

        # 1) Preprocess
        self.ensure_preprocess()

        # 2) Treino incremental
        self.train()

        # 3) Backtest
        self.run_backtest()

        # 4) Inferência opcional
        if do_inference:
            preds = self.inference()
            print("Predições finais:", preds)

        print("\n===================================================")
        print("🎯 AUTO ML — PIPELINE COMPLETO FINALIZADO")
        print("===================================================\n")


# ======================================================================
# EXECUÇÃO DIRETA
# ======================================================================
if __name__ == "__main__":
    master = AutoMLMaster(
        ticker="GALP.LS",
        window_size=60,
        horizon=3,
        frequency="1h",
        model_name="TCN"
    )

    master.run_all(do_inference=True)
