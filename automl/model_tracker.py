# ======================================================================
#  model_tracker.py — Sistema industrial de tracking de modelos
#  Guarda histórico de versões, métricas, backtests e seleção automática.
# ======================================================================

import json
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd


class ModelTracker:
    """
    Sistema central de rastreamento e memória dos modelos.
    Guarda tudo o que um ambiente AutoML evolutivo precisa:
        - versões
        - treino / validação
        - métricas de backtest
        - hiperparâmetros
        - timestamps
    """

    def __init__(self, model_dir: Path):
        """
        model_dir é a pasta do ticker:
            ex: storage/models/GALP.LS/
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.model_dir / "model_history.json"
        self.best_path = self.model_dir / "best_model.json"
        self.metrics_path = self.model_dir / "metrics.csv"

        # inicializar ficheiros caso não existam
        if not self.log_path.exists():
            self._write_json(self.log_path, {"versions": []})

        if not self.metrics_path.exists():
            df = pd.DataFrame(
                columns=[
                    "version", "timestamp",
                    "train_loss", "val_loss",
                    "sharpe", "calmar", "max_drawdown",
                    "winrate", "total_return",
                    "hyperparams"
                ]
            )
            df.to_csv(self.metrics_path, index=False)

    # --------------------------------------------------------------
    # Internos utilitários
    # --------------------------------------------------------------
    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    # --------------------------------------------------------------
    # Criar nova versão
    # --------------------------------------------------------------
    def register_new_version(self) -> int:
        history = self._read_json(self.log_path)
        versions = history["versions"]

        new_version = len(versions) + 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "version": new_version,
            "timestamp": timestamp,
            "train_loss": None,
            "val_loss": None,
            "metrics": None,
            "hyperparams": None
        }

        versions.append(record)
        history["versions"] = versions
        self._write_json(self.log_path, history)

        return new_version

    # --------------------------------------------------------------
    # Guardar métricas de treino
    # --------------------------------------------------------------
    def record_training_metrics(self, version: int, train_loss: float, val_loss: float):
        history = self._read_json(self.log_path)

        for v in history["versions"]:
            if v["version"] == version:
                v["train_loss"] = train_loss
                v["val_loss"] = val_loss

        self._write_json(self.log_path, history)

    # --------------------------------------------------------------
    # Guardar métricas de backtest
    # --------------------------------------------------------------
    def record_backtest_metrics(self, version: int, metrics: Dict[str, Any]):
        history = self._read_json(self.log_path)

        for v in history["versions"]:
            if v["version"] == version:
                v["metrics"] = metrics

        self._write_json(self.log_path, history)

        # actualizar CSV
        df = pd.read_csv(self.metrics_path)

        new_row = {
            "version": version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": None,
            "val_loss": None,
            "sharpe": metrics.get("sharpe"),
            "calmar": metrics.get("calmar"),
            "max_drawdown": metrics.get("max_drawdown"),
            "winrate": metrics.get("winrate"),
            "total_return": metrics.get("total_return"),
            "hyperparams": None
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.metrics_path, index=False)

    # --------------------------------------------------------------
    # Guardar hiperparâmetros
    # --------------------------------------------------------------
    def record_hyperparams(self, version: int, hyperparams: Dict[str, Any]):
        history = self._read_json(self.log_path)

        for v in history["versions"]:
            if v["version"] == version:
                v["hyperparams"] = hyperparams

        self._write_json(self.log_path, history)

    # --------------------------------------------------------------
    # Selecionar melhor modelo
    # --------------------------------------------------------------
    def update_best_model(self):
        history = self._read_json(self.log_path)
        versions = history["versions"]

        if not versions:
            return None

        # critério: menor val_loss
        best = min(
            [v for v in versions if v["val_loss"] is not None],
            key=lambda x: x["val_loss"],
        )

        self._write_json(self.best_path, best)
        return best

    # --------------------------------------------------------------
    # Obter melhor modelo
    # --------------------------------------------------------------
    def load_best_model_metadata(self):
        if not self.best_path.exists():
            return None
        return self._read_json(self.best_path)
