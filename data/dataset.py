import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Dataset temporal simples e robusto.

    Recebe:
    - X: janelas (numPy array) com shape [N, window_size, features]
    - y: targets com shape [N]
    """

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("TimeSeriesDataset: X e y têm comprimentos diferentes.")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModule:
    """
    Cria DataLoaders para treino e validação.
    - shuffle só no treino
    - mantém ordem temporal na validação
    """

    def __init__(
        self,
        X,
        y,
        batch_size=32,
        val_pct=0.2,
    ):
        if not 0 < val_pct < 1:
            raise ValueError("DataModule: val_pct deve estar entre 0 e 1.")

        self.batch_size = batch_size
        self.val_pct = val_pct

        total = len(X)
        val_size = int(total * val_pct)
        train_size = total - val_size

        self.X_train = X[:train_size]
        self.y_train = y[:train_size]

        self.X_val = X[train_size:]
        self.y_val = y[train_size:]

    def get_dataloaders(self):
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val)

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,   # nunca embaralhar validação
            drop_last=False,
        )

        return train_dl, val_dl

    def get_inference_loader(self):
        """Loader de inferência (sem shuffle)."""
        dataset = TimeSeriesDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=1, shuffle=False)
