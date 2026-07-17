"""Fetches everything a Docker build needs to reproduce a registered model's 'champion' (or
other alias) version -- pyproject.toml, uv.lock, the training/inference scripts, and config/ --
from the run that produced it. This is deliberately the *only* thing that needs to exist
locally besides the Dockerfile itself: run this against any machine's MLflow server and it
reconstructs the same directory shape (scripts/ sibling to config/) the original repo has,
with nothing else checked out.

Usage:
    python fetch_artifacts.py --model-name mock-model --alias champion --dest /app
"""
import argparse
import os
import mlflow
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="mock-model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--dest", default="/app")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(args.model_name, args.alias)
    run_id = model_version.run_id
    assert run_id is not None, f"model version {model_version.version} has no associated run_id"

    os.makedirs(args.dest, exist_ok=True)
    scripts_dir = os.path.join(args.dest, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    # repo-root-equivalent files: uv looks for these directly in the project root
    for filename in ("pyproject.toml", "uv.lock"):
        client.download_artifacts(run_id, filename, args.dest)

    # scripts/ -- mirrors the local layout so `from mock_train_torch import ...` (same
    # directory) and mock_inference_torch.py's Hydra config_path="../config" (one level up)
    # both resolve exactly the way they do in the original repo
    for filename in ("mock_train_torch.py", "mock_inference_torch.py"):
        client.download_artifacts(run_id, filename, scripts_dir)

    # config/ -- downloaded as a whole directory, not a single file
    client.download_artifacts(run_id, "config", args.dest)

    print(f"Fetched pyproject.toml, uv.lock, scripts/, config/ from run {run_id} "
          f"(model '{args.model_name}@{args.alias}', version {model_version.version}) into {args.dest}")


if __name__ == "__main__":
    main()
