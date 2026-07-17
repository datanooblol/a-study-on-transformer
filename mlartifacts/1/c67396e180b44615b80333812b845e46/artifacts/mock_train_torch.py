import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class SimpleNetwork(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList([
            *args
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)   # without this, two stacked linear layers collapse into one -- see LinearLayer note below
        return x

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def generate_dataset(n:int=200, d_model:int=2):
    x = torch.randn(n, d_model, dtype=torch.float)
    y = torch.randn(n, 1, dtype=torch.float)
    return x, y

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss) -> bool:
        """Returns True if val_loss is a new best."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

@hydra.main(version_base=None, config_path="../config", config_name="config")
def my_app(cfg : DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = generate_dataset(n=cfg.params.n, d_model=cfg.params.d_model)
    train_dataset = SimpleDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    x_val,  y_val = generate_dataset(n=cfg.params.n, d_model=cfg.params.d_model)
    val_dataset = SimpleDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=cfg.params.batch_size, shuffle=False)
    layer1 = LinearLayer(cfg.params.d_model, cfg.params.expansion*cfg.params.d_model)
    layer2 = LinearLayer(cfg.params.expansion*cfg.params.d_model, 1)
    model = SimpleNetwork(layer1, layer2)
    model.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=cfg.params.lr)

    early_stopper = EarlyStopping(patience=5)
    input_example = x_train[:1].to(device)   # one real sample -- MLflow traces model.forward on this to serialize it

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg.params, resolve=True))

        # log the exact dependency manifest + this script's own source, so a deployment target can
        # rebuild the identical environment via `uv sync` instead of trusting MLflow's auto-inferred
        # requirements. get_original_cwd() is required here because Hydra changes the working
        # directory to its own output folder by default -- a plain relative path would look for
        # these files in the wrong place.
        repo_root = get_original_cwd()
        mlflow.log_artifact(os.path.join(repo_root, "pyproject.toml"))
        mlflow.log_artifact(os.path.join(repo_root, "uv.lock"))
        mlflow.log_artifact(os.path.abspath(__file__))   # this file itself -- LinearLayer/SimpleNetwork live here
        mlflow.log_artifact(os.path.join(repo_root, "scripts", "mock_inference_torch.py"))
        mlflow.log_artifacts(os.path.join(repo_root, "config"), artifact_path="config")   # whole directory, not just config.yaml -- future config files come along automatically

        for epoch in range(cfg.params.epochs):
            model.train()
            total_train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(x_batch)
                loss = F.mse_loss(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    output = model(x_batch)
                    loss = F.mse_loss(output, y_batch)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            # log every epoch regardless of print cadence -- the MLflow UI curve needs full resolution
            mlflow.log_metrics(
                {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch
            )

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{cfg.params.epochs}], '
                      f'train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')

            improved = early_stopper.step(avg_val_loss)
            if improved:
                # re-logs to the same artifact path each time -- always holds the best model seen so far, not one per epoch
                mlflow.pytorch.log_model(model, name="best_model", input_example=input_example, serialization_format="pickle")  # type: ignore[arg-type] -- MLflow's own runtime docs confirm torch.Tensor is valid; the stub's ModelInputExample union just wasn't widened for it

            if early_stopper.should_stop:
                print(f'Early stopping at epoch {epoch+1} (best val_loss={early_stopper.best_loss:.4f})')
                break

        mlflow.pytorch.log_model(model, name="final_model", input_example=input_example, serialization_format="pickle")  # type: ignore[arg-type]

        # register exactly once per run, from the best checkpoint (not final_model, which may be
        # worse than best_model if early stopping already fired) -- registering inside the
        # `if improved:` block instead would create a noisy new version on every single improvement
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(model_uri=f"runs:/{run_id}/best_model", name="mock-model")

# dependencies = [
#   "hydra-core>=1.3.4",
#   "matplotlib>=3.11.0",
#   "numpy>=2.4.4",
#   "pandas>=3.0.2",
#   "plotly>=6.7.0",
#   "scikit-learn>=1.8.0",
#   "torch>=2.11.0",
#   "umap-learn>=0.5.12",
# ]
if __name__ == "__main__":
    my_app()
