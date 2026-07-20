import os
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import torch

# Unused directly below, but required: the model was pickled with these exact classes, and
# pickle resolves them by looking up `__main__.LinearLayer` / `__main__.SimpleNetwork` at load
# time. Importing them here makes them attributes of *this* script's module -- which becomes
# `__main__` when run directly -- so unpickling finds them regardless of which file originally
# defined them.
from mock_train_torch import LinearLayer, SimpleNetwork, generate_dataset  # noqa: F401

MODEL_URI = "models:/mock-model@champion"

@hydra.main(version_base=None, config_path="../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # MLFLOW_TRACKING_URI wins if set (e.g. a real server URL inside a container) -- falls back
    # to the local sqlite file from config.yaml for plain local runs
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mlflow.pytorch.load_model(MODEL_URI)
    model.to(device)
    model.eval()

    # this toy dataset's x/y are independent random noise (see generate_dataset in mock_train_torch.py) --
    # predictions here reflect whatever the model learned from noise, not a real relationship
    x_sample, _ = generate_dataset(n=5, d_model=cfg.params.d_model)
    x_sample = x_sample.to(device)

    with torch.no_grad():
        predictions = model(x_sample)

    for i, (x, pred) in enumerate(zip(x_sample, predictions)):
        print(f"Sample {i}: input={x.cpu().tolist()}, prediction={pred.item():.4f}")

if __name__ == "__main__":
    my_app()
