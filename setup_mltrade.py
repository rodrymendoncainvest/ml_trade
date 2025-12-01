"""
setup_mltrade.py — Script de setup completo para o backend ML_Trade

Este script:
- verifica Python >= 3.10
- cria ambiente virtual .venv se não existir
- instala dependências reais do projeto
- cria a estrutura storage/
- cria .env (se não existir)
- valida imports principais
- corre um smoke test da pipeline (opcional)

Importante:
Corre este script na raiz do projeto ML_Trade:
    python setup_mltrade.py
"""

import os
import sys
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------
# 1. Validação da versão do Python
# ---------------------------------------------------------------------
def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        print("[❌] Precisas de Python 3.10 ou superior.")
        sys.exit(1)
    print(f"[✔] Python {major}.{minor} OK")


# ---------------------------------------------------------------------
# 2. Criar ambiente virtual
# ---------------------------------------------------------------------
def create_venv():
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        print("[⚙] A criar ambiente virtual .venv...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    print("[✔] Ambiente virtual OK")


# ---------------------------------------------------------------------
# 3. Instalar dependências
# ---------------------------------------------------------------------
def install_deps():
    reqs = [
        "numpy",
        "pandas",
        "yfinance",
        "torch",
        "statsmodels",
        "joblib",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "requests",
        "python-dotenv"
    ]

    pip_path = Path(".venv/Scripts/pip.exe") if os.name == "nt" else Path(".venv/bin/pip")

    print("[⚙] A instalar dependências...")
    subprocess.run([str(pip_path), "install"] + reqs, check=True)
    print("[✔] Dependências instaladas")


# ---------------------------------------------------------------------
# 4. Criar estrutura ML_Trade/storage
# ---------------------------------------------------------------------
def create_structure():
    from config.paths import PATHS

    for key, p in PATHS.items():
        os.makedirs(p, exist_ok=True)

    print("[✔] Pastas do ML_Trade criadas:")
    for key, p in PATHS.items():
        print(f"   - {key}: {p}")


# ---------------------------------------------------------------------
# 5. Criar ficheiro .env se não existir
# ---------------------------------------------------------------------
def create_env():
    env_path = Path(".env")
    if not env_path.exists():
        print("[⚙] A criar .env...")
        env_path.write_text(
            "#############################\n"
            "# ML_Trade .env (vazio)\n"
            "#############################\n\n"
            "# --- Providers ---\n"
            "EODHD_KEY=\n"
            "TWELVE_DATA_KEY=\n"
            "NEWSAPI_KEY=\n"
            "FINNHUB_KEY=\n"
            "FMP_KEY=\n"
            "ALPHAVANTAGE_KEY=\n"
            "POLYGON_KEY=\n\n"
            "# --- System overrides ---\n"
            "ML_TRADE_WINDOW_SIZE=\n"
            "ML_TRADE_HORIZON=\n"
            "ML_TRADE_BATCH_SIZE=\n"
        )
        print("[✔] .env criado (vazio)")
    else:
        print("[✔] .env já existe — não mexi")


# ---------------------------------------------------------------------
# 6. Smoke test dos imports principais
# ---------------------------------------------------------------------
def smoke_test():
    print("[⚙] A validar módulos ML_Trade...")

    try:
        import config.settings
        import config.paths
        import data.raw_ingestor
        import data.validator
        import data.aligner
        import data.translator
        import data.features

        print("[✔] Imports principais OK")
    except Exception as e:
        print("[❌] Erro de import:", e)
        sys.exit(1)


# ---------------------------------------------------------------------
# 7. Pipeline test opcional
# ---------------------------------------------------------------------
def test_pipeline():
    print("[⚙] A correr pipeline de teste (sem treino)...")

    try:
        from pipeline import run_pipeline
        run_pipeline("AAPL", window_size=32, horizon=1)
        print("[✔] Pipeline executou com sucesso")
    except Exception as e:
        print("[❌] Erro na pipeline:", e)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("--------------------------------------------")
    print("        ML_Trade — Setup Industrial         ")
    print("--------------------------------------------")

    check_python()
    create_venv()
    install_deps()
    create_structure()
    create_env()
    smoke_test()

    print("\n[ℹ] Setup concluído com sucesso.")

    # Descomenta se quiseres correr a pipeline automaticamente:
    # test_pipeline()
