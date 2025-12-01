# models/registry.py

from models.networks.tcn import TCNModel
from models.networks.lstm import LSTMModel
from models.networks.gru import GRUModel
from models.networks.transformer import TransformerModel
from models.networks.nbeats import NBeatsModel


class ModelRegistry:
    """
    Registry industrial dos modelos suportados.
    Permite criar modelos apenas pelo nome.
    """

    _models = {
        "TCN": TCNModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "TRANSFORMER": TransformerModel,
        "NBEATS": NBeatsModel,
    }

    @staticmethod
    def list():
        return list(ModelRegistry._models.keys())

    @staticmethod
    def create(name: str, **kwargs):
        name = name.upper()

        if name not in ModelRegistry._models:
            raise ValueError(
                f"Modelo '{name}' não existe. "
                f"Modelos disponíveis: {list(ModelRegistry._models.keys())}"
            )

        return ModelRegistry._models[name](**kwargs)
