"""
constants.py — Constantes globais e imutáveis do backend ML_Trade.

Aqui ficam:
- limites numéricos
- valores estruturais que nunca mudam em runtime
- nomes de colunas reservadas
- epsilons numéricos
- thresholds universais
"""

import numpy as np

# ----------------------------------------------------------------------
# Numéricos fundamentais
# ----------------------------------------------------------------------

# epsilon numérico para evitar divisões por zero e log de zero
EPSILON = 1e-9

# limite máximo de outliers permitido antes de invalidação
MAX_OUTLIER_FACTOR = 10   # usado em validação de amplitudes OHLC

# limite mínimo de candles após limpeza de features
MIN_FEATURE_HISTORY = 200

# tamanho mínimo de dataset para treino (além do min_history_points do pipeline)
MIN_TRAIN_SAMPLES = 500

# limite máximo de velas sintéticas permitidas após aligner
MAX_SYNTHETIC_PERCENT = 0.05   # 5%

# ----------------------------------------------------------------------
# Estrutura de colunas
# ----------------------------------------------------------------------

# colunas OHLCV básicas (ordem universal)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# colunas adicionais internas
INTERNAL_COLUMNS = [
    "timestamp",
    "synthetic",
    "logreturn"
]

# colunas proibidas para modelos (não devem ser features)
FORBIDDEN_FEATURES = [
    "timestamp",
    "synthetic"
]

# ----------------------------------------------------------------------
# Indicadores técnicos — defaults estruturais
# ----------------------------------------------------------------------

# períodos padrão de indicadores
DEFAULT_RSI_PERIOD = 14
DEFAULT_ATR_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

# ----------------------------------------------------------------------
# Valores estruturais do sistema
# ----------------------------------------------------------------------

# amostras mínimas necessárias para qualquer indicador técnico funcionar corretamente
MIN_TECHNICAL_LOOKBACK = 52   # suficiente para Ichimoku

# taxa mínima de autocorrelação aceitável para validar uma série temporal
MIN_AUTOCORR_THRESHOLD = 0.05

# número máximo de velas seguidas com volume zero considerado normal
MAX_VOLUME_ZERO_RUN = 10

# ----------------------------------------------------------------------
# Seeds e reprodutibilidade
# ----------------------------------------------------------------------

# seed global (pode ser usado no trainer para fixar resultados)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
