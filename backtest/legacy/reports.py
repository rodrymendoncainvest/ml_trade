import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BacktestReport:
    """
    Report industrial para backtesting de modelos ML.
    Produz:
        - TXT (relatório humano)
        - JSON (estrutura para API/Dashboard)
        - CSV (resultados completos)
        - PNGs (equity, drawdown, distribuição)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # AUXILIAR: PLOT EQUITY
    # ==================================================================
    def plot_equity(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 5))
        plt.plot(df["timestamp"], df["equity"], label="Equity", color="blue")
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()

        out = self.output_dir / "equity_curve.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    # ==================================================================
    # AUXILIAR: PLOT DRAWDOWN
    # ==================================================================
    def plot_drawdown(self, df: pd.DataFrame):
        equity = df["equity"]
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max

        plt.figure(figsize=(12, 4))
        plt.fill_between(df["timestamp"], dd, 0, color="red", alpha=0.3)
        plt.title("Drawdown Curve")
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.grid(True)

        out = self.output_dir / "drawdown_curve.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    # ==================================================================
    # AUXILIAR: DISTRIBUIÇÃO DE RETORNOS
    # ==================================================================
    def plot_return_distribution(self, df: pd.DataFrame):
        returns = df["pnl"].replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(10, 5))
        plt.hist(returns, bins=50, color="darkgreen", alpha=0.7)
        plt.title("Return Distribution")
        plt.xlabel("PNL")
        plt.ylabel("Frequency")
        plt.grid(True)

        out = self.output_dir / "return_distribution.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        return out

    # ==================================================================
    # ANÁLISE DE TRADES
    # ==================================================================
    def summarize_trades(self, df: pd.DataFrame):
        signals = df["signal"]
        close = df["close"]

        trades = []
        position = 0
        entry_price = None

        for i in range(1, len(df)):
            if signals[i] != position:
                # Fechar
                if position != 0 and entry_price is not None:
                    pnl = (close[i] - entry_price) * position
                    trades.append(pnl)

                # Abrir
                if signals[i] != 0:
                    entry_price = close[i]

            position = signals[i]

        return {
            "num_trades": len(trades),
            "avg_pnl": float(np.mean(trades)) if trades else 0.0,
            "winrate": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
            "long_trades": int(sum(1 for t in trades if t > 0)),
            "short_trades": int(sum(1 for t in trades if t < 0)),
            "trade_list": trades,
        }

    # ==================================================================
    # CRIAR TXT (leitura humana)
    # ==================================================================
    def write_txt(self, metrics: dict, trade_info: dict):
        path = self.output_dir / "report.txt"

        with open(path, "w") as f:
            f.write("=============================================\n")
            f.write("        BACKTEST REPORT (TXT VERSION)\n")
            f.write("=============================================\n\n")

            f.write("---- PERFORMANCE ----\n")
            for k, v in metrics.items():
                f.write(f"{k:20}: {v}\n")

            f.write("\n---- TRADES ----\n")
            for k, v in trade_info.items():
                f.write(f"{k:20}: {v}\n")

        return path

    # ==================================================================
    # CRIAR JSON (estruturado, para API)
    # ==================================================================
    def write_json(self, metrics: dict, trade_info: dict):
        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "performance": metrics,
                    "trades": trade_info,
                },
                f,
                indent=4
            )
        return path

    # ==================================================================
    # SALVAR CSV DE RESULTADOS
    # ==================================================================
    def save_results_csv(self, df: pd.DataFrame):
        path = self.output_dir / "results.csv"
        df.to_csv(path, index=False)
        return path

    # ==================================================================
    # REPORT PRINCIPAL
    # ==================================================================
    def generate_report(self, df: pd.DataFrame, metrics: dict, save_plots=True):
        """
        Gera relatórios completos em:
          - TXT
          - JSON
          - CSV
          - PNGs (se save_plots=True)
        """

        trade_info = self.summarize_trades(df)

        # salvar resultados CSV
        csv_path = self.save_results_csv(df)

        # gerar plots
        plots = {}
        if save_plots:
            plots["equity"] = self.plot_equity(df)
            plots["drawdown"] = self.plot_drawdown(df)
            plots["return_distribution"] = self.plot_return_distribution(df)

        # gerar TXT e JSON
        txt_path = self.write_txt(metrics, trade_info)
        json_path = self.write_json(metrics, trade_info)

        return {
            "csv": csv_path,
            "txt": txt_path,
            "json": json_path,
            "plots": plots,
        }
