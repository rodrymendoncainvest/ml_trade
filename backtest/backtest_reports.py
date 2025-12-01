import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import pandas as pd

from .backtest_metrics import BacktestMetrics


class BacktestReport:
    """
    BacktestReport V2 — industrial, completo e profissional.

    Gera:
        - equity_curve.png
        - drawdown.png
        - signals_comparison.png
        - metrics.txt (todas as métricas da avaliação)
    """

    def __init__(self, out_dir: Path, ticker: str):
        self.out_dir = Path(out_dir)
        self.ticker = ticker.upper()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # 1) EQUITY CURVE
    # ===================================================================
    def save_equity_plot(self, df: pd.DataFrame):
        plt.figure(figsize=(14, 6))
        plt.plot(df["equity_curve"], label="Equity Curve", linewidth=1.6)
        plt.title(f"Equity Curve — {self.ticker}", fontsize=15)
        plt.xlabel("Index")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = self.out_dir / f"{self.ticker}_equity.png"
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    # ===================================================================
    # 2) DRAWDOWN PLOT
    # ===================================================================
    def save_drawdown_plot(self, df: pd.DataFrame):
        mdd, dd_series = BacktestMetrics.max_drawdown(df["equity_curve"])

        plt.figure(figsize=(14, 6))
        plt.plot(dd_series, color="red", linewidth=1)
        plt.title(f"Drawdown — {self.ticker}", fontsize=15)
        plt.xlabel("Index")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        path = self.out_dir / f"{self.ticker}_drawdown.png"
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    # ===================================================================
    # 3) COMPARAÇÃO DE SINAIS (Técnico vs ML vs Final)
    # ===================================================================
    def save_signal_comparison(self, df: pd.DataFrame):
        """
        Requer colunas:
            - ml_signal (opcional)
            - tech_signal (opcional)
            - signal (final obrigatório)
        """
        plt.figure(figsize=(16, 7))

        # Não quebramos se faltar uma das colunas
        if "ml_signal" in df.columns:
            plt.plot(df["ml_signal"], label="ML Signal", alpha=0.6)
        if "tech_signal" in df.columns:
            plt.plot(df["tech_signal"], label="Technical Signal", alpha=0.6)

        plt.plot(df["signal"], label="Final Signal", linewidth=2)

        plt.title(f"Signal Comparison — {self.ticker}", fontsize=15)
        plt.xlabel("Index")
        plt.ylabel("Signal (-1, 0, +1)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = self.out_dir / f"{self.ticker}_signals.png"
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    # ===================================================================
    # 4) TEXTO DE MÉTRICAS COMPLETO
    # ===================================================================
    def save_metrics_txt(self, metrics: Dict):
        path = self.out_dir / f"{self.ticker}_metrics.txt"

        with open(path, "w") as f:
            f.write("BACKTEST METRICS V2\n")
            f.write("====================\n\n")

            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        return path

    # ===================================================================
    # 5) EXECUTAR TODOS OS RELATÓRIOS NUMA CHAMADA
    # ===================================================================
    def generate_full_report(self, df: pd.DataFrame, metrics: Dict):
        """
        Cria todos os gráficos e o txt de métricas.
        """
        self.save_equity_plot(df)
        self.save_drawdown_plot(df)
        self.save_signal_comparison(df)
        self.save_metrics_txt(metrics)
