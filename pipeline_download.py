# ============================================================
# pipeline_download.py — Download automático de dados 1H
# ============================================================

import argparse
from data.data_downloader import DataDownloader


def run_download(ticker: str):
    print("-----------------------------------------------------")
    print(f"DOWNLOAD PIPELINE — {ticker}")
    print("-----------------------------------------------------\n")

    downloader = DataDownloader(period="730d")
    downloader.download_1h(ticker)

    print("\nDOWNLOAD CONCLUÍDO")
    print("-----------------------------------------------------\n")


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 1H data for a ticker.")
    parser.add_argument("--ticker", type=str, required=True)

    args = parser.parse_args()

    run_download(args.ticker)
