import os
import pandas as pd


class RawIngestor:
    """
    Ingestor RAW:
    - lê CSVs já descarregados em data/history/<TICKER>_1H.csv
    - converte 'Datetime' -> 'timestamp'
    - garante formato consistente para o pipeline
    """

    def __init__(self):
        # Diretório local dos históricos
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "history")
        )

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Lê um CSV produzido pelo data_downloader.
        filename pode ser:
            "data/history/GALP.LS_1H.csv"
        ou apenas:
            "GALP.LS_1H.csv"
        """

        # Ajustar caminho se vier com prefixo "data/history/"
        if not os.path.exists(filename):
            filename = os.path.join(self.data_dir, os.path.basename(filename))

        if not os.path.exists(filename):
            raise FileNotFoundError(f"RawIngestor: ficheiro não encontrado → {filename}")

        df = pd.read_csv(filename)

        # ------------------------------------------------------
        # Corrigir nome da coluna: Datetime -> timestamp
        # ------------------------------------------------------
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "timestamp" not in df.columns:
            raise ValueError("RawIngestor: CSV não tem coluna Datetime nem timestamp.")

        # Converter timestamp para datetime UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Remover linhas com timestamp inválido
        df = df.dropna(subset=["timestamp"])

        # Ordenar
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df
