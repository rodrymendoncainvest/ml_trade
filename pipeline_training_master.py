# pipeline_training_master.py
from automl.model_selector import ModelSelector


if __name__ == "__main__":
    ticker = "GALP.LS"

    selector = ModelSelector(
        ticker=ticker,
        epochs=40,
        lr=1e-3,
        batch_size=64
    )

    selector.run()
