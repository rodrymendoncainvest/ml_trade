# registry.py
import torch
from typing import Dict, Callable

# Importar modelos
from models.networks.tcn import TCNModel
from models.networks.gru import GRUModel
from models.networks.transformer import TransformerModel
from models.networks.nbeats import NBeatsModel


class ModelRegistry:
    """
    Model Registry Industrial V1
    -----------------------------------
    - Carrega modelos por string (ex: "tcn", "gru", "transformer", "nbeats")
    - Mantém fábrica centralizada
    - Simplifica o pipeline de treino e inferência
    - Permite adicionar novos modelos sem alterar o resto do sistema
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

        # ---- Registo inicial de modelos ----
        self.register("tcn", TCNModel)
        self.register("gru", GRUModel)
        self.register("transformer", TransformerModel)
        self.register("nbeats", NBeatsModel)

    # ----------------------------------------------------------
    # REGISTAR MODELO
    # ----------------------------------------------------------
    def register(self, name: str, constructor: Callable):
        """
        Adiciona um novo modelo ao registry.
        """
        name = name.lower()

        if name in self._registry:
            raise ValueError(f"[ModelRegistry] Modelo '{name}' já existe.")

        self._registry[name] = constructor

    # ----------------------------------------------------------
    # CONSTRUIR MODELO
    # ----------------------------------------------------------
    def create(self, name: str, **kwargs):
        """
        Cria um modelo a partir do nome.

        Ex:
            model = registry.create(
                "tcn",
                input_dim=n_features,
                window_size=60,
                horizon=3
            )
        """
        name = name.lower()

        if name not in self._registry:
            raise ValueError(
                f"[ModelRegistry] Modelo '{name}' não encontrado. "
                f"Disponíveis: {list(self._registry.keys())}"
            )

        model_class = self._registry[name]
        model = model_class(**kwargs)

        return model

    # ----------------------------------------------------------
    # LISTAR MODELOS
    # ----------------------------------------------------------
    def available_models(self):
        return list(self._registry.keys())


# Instância global
registry = ModelRegistry()
