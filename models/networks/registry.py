"""
ModelRegistry — Fábrica Industrial de Modelos ML_Trade (VERSÃO FINAL)
---------------------------------------------------------------------

Este módulo gere a criação de todos os modelos disponíveis no sistema:

    - TCN
    - LSTM
    - GRU
    - TRANSFORMER
    - NBEATS  (exige window_size)

O objetivo é centralizar toda a lógica de criação e evitar erros no pipeline.
"""

from models.networks.tcn import TCNModel
from models.networks.lstm import LSTMModel
from models.networks.gru import GRUModel
from models.networks.transformer import TransformerModel
from models.networks.nbeats import NBeatsModel


class ModelRegistry:
    """
    Registry industrial dos modelos suportados.

    create(model_name, **kwargs) trata automaticamente:
        - parâmetros obrigatórios
        - APIs específicas de cada modelo
        - coerência entre assinatura e pipeline
    """

    # Modelos disponíveis
    _models = {
        "TCN": TCNModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "TRANSFORMER": TransformerModel,
        "NBEATS": NBeatsModel,
    }

    @staticmethod
    def list():
        """Lista todos os modelos disponíveis."""
        return list(ModelRegistry._models.keys())

    @staticmethod
    def create(name: str, **kwargs):
        """
        Cria um modelo pelo nome.

        Parâmetros esperados:
            - input_dim: obrigatório para todos
            - horizon:   obrigatório para todos
            - window_size: obrigatório apenas para NBEATS
        """

        name = name.upper()

        if name not in ModelRegistry._models:
            raise ValueError(
                f"Modelo '{name}' não existe. "
                f"Disponíveis: {list(ModelRegistry._models.keys())}"
            )

        ModelClass = ModelRegistry._models[name]

        # ------------------------------------------------------------------
        # NBEATS → requer explicitamente window_size
        # ------------------------------------------------------------------
        if name == "NBEATS":
            if "window_size" not in kwargs:
                raise ValueError(
                    "NBEATS requer argumento obrigatório 'window_size'. "
                    "Certifica-te que passas o mesmo window_size usado no dataset."
                )

            return ModelClass(
                input_dim=kwargs["input_dim"],
                window_size=kwargs["window_size"],
                horizon=kwargs.get("horizon", 1),
            )

        # ------------------------------------------------------------------
        # OUTROS MODELOS — criação simples
        # ------------------------------------------------------------------
        return ModelClass(
            input_dim=kwargs["input_dim"],
            horizon=kwargs.get("horizon", 1),
        )
