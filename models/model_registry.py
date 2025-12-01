# model_registry.py
"""
Registo central de modelos do ML_Trade.
Permite carregar qualquer modelo por string ("tcn", "lstm", ...)
sem if/else espalhados pelo projeto.

Cada modelo deve implementar:
    - __init__(input_dim, hidden_dim, output_dim, ...)
    - forward(x)
    - predict(x)
"""

from models.networks.tcn import TCNModel
from models.networks.lstm import LSTMModel
from models.networks.gru import GRUModel
from models.networks.transformer import TransformerModel
from models.networks.nbeats import NBeatsModel


MODEL_REGISTRY = {
    "tcn": TCNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TransformerModel,
    "nbeats": NBeatsModel,
}


def get_model_class(name: str):
    """
    Retorna a classe do modelo registado.
    """
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{name}' não está registado. "
            f"Modelos disponíveis: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def create_model(name: str, **kwargs):
    """
    Instancia um modelo usando o nome registado.
    Exemplo:
        model = create_model("tcn", input_dim=64, output_dim=1)
    """
    cls = get_model_class(name)
    return cls(**kwargs)
