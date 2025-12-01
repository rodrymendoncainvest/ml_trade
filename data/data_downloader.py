# data_downloader.py — versão final (Yahoo + limpeza automática)
import os
import pandas as pd
import yfinance as yf


# Diretório onde os históricos ficam guardados
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "history")
)
os.makedirs(DATA_DIR, exist_ok=True)


class DataDownloader:
    """
    Downloader 1H robusto:
    - descarrega via Yahoo Finance
    - normaliza colunas OHLCV
    - força timestamps para UTC
    - remove velas 00:00 e fora da janela de mercado óbvia
    - remove duplicados
    - remove NaNs estruturais
    """

    def __init__(self, period="730d"):
        self.period = period

    # ------------------------------------------------------------------
    def download_1h(self, ticker: str) -> pd.DataFrame:
        """
        Descarrega candles de 1H e limpa-os.
        Guarda em data/history/<TICKER>_1H.csv
        """

        print(f"[Downloader] A descarregar {ticker} 1H...")

        df = yf.download(
            tickers=ticker,
            interval="1h",
            period=self.period,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise RuntimeError(f"[Downloader] Yahoo devolveu dataset vazio para {ticker}")

        # --------------------------------------------------------------
        # NORMALIZAÇÃO DE COLUNAS
        # --------------------------------------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        keep = ["open", "high", "low", "close", "volume"]
        df = df[keep]

        # --------------------------------------------------------------
        # GARANTIR TIMESTAMP UTC
        # --------------------------------------------------------------
        idx = df.index

        try:
            if idx.tz is None:
                df.index = idx.tz_localize("UTC")
            else:
                df.index = idx.tz_convert("UTC")
        except Exception:
            df.index = pd.to_datetime(idx, utc=True, errors="coerce")

        df = df.dropna()
        df = df.sort_index()

        # --------------------------------------------------------------
        # REMOVER VELAS 00:00 e 23:00 (que são lixo em equities Yahoo)
        # --------------------------------------------------------------
        df["hour"] = df.index.hour
        df = df[(df["hour"] >= 1) & (df["hour"] <= 22)]
        df = df.drop(columns=["hour"])

        # --------------------------------------------------------------
        # REMOVER DUPLICADOS DE TIMESTAMP
        # --------------------------------------------------------------
        df = df[~df.index.duplicated(keep="first")]

        # --------------------------------------------------------------
        # SAVE
        # --------------------------------------------------------------
        out_path = os.path.join(DATA_DIR, f"{ticker}_1H.csv")
        df.to_csv(out_path, index=True)

        print(f"[Downloader] Guardado em: {out_path}")
        print(df.head())

        return df
