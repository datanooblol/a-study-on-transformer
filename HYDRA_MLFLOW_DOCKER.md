# Hydra + MLflow + Docker — Runbook

Practical, copy-pasteable commands for the full pipeline: start the tracking server, train, verify/promote
a model, run inference, then build and run the Docker image. For *why* each step is shaped this way — the
serialization format choice, the backend/artifact store split, Docker networking, `uv` reproducibility, Hydra's
`config_path` resolution — see [`lessons/27-mlflow-deployment-pipeline.html`](lessons/27-mlflow-deployment-pipeline.html).
This file is the "what to type," that lesson is the "why it has to be typed that way."

Everything below assumes commands are run from the repo root (`D:\a-study-on-transformer`), with `uv sync`
already run once to set up `.venv`.

## Pipeline overview

```
mlflow server (own terminal, stays running)
      ↓
train (registers a new "mock-model" version, doesn't move "champion")
      ↓
reassign "champion" to the new version (manual, always)
      ↓
run inference locally, as a sanity check
      ↓
docker build (fetches pyproject.toml/uv.lock/scripts/config from that run, syncs an identical env)
      ↓
docker run (loads "champion", runs inference in the container)
```

## 1. Start the MLflow server

```
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000 --allowed-hosts "host.docker.internal:5000,localhost:5000,127.0.0.1:5000"
```

Leave this running in its own terminal — everything else below talks to it. `--artifacts-destination` (not
`--default-artifact-root`) is required for the Docker build to be able to fetch model/config artifacts later;
`--allowed-hosts` is required for Docker to be able to reach it at all. UI: `http://127.0.0.1:5000`.

## 2. Train

```
uv run scripts\mock_train_torch.py
```

Each run: logs params/metrics, logs `pyproject.toml`/`uv.lock`/both scripts/`config/` as artifacts, logs
`best_model` and `final_model` (with `best_val_loss`/`final_val_loss`/`best_epoch` metrics), tags the run with
`stage`, and registers a **new** `mock-model` version from `best_model`.

Override any config value from the command line without editing `config/config.yaml`:

```
uv run scripts\mock_train_torch.py run_name="lr-sweep-1" stage="experiment" params.lr=0.01 params.epochs=200
```

## 3. Reassign the `champion` alias

Registering a new version **never** moves `champion` automatically — this is a separate, deliberate step,
every single time:

```python
from mlflow import MlflowClient
MlflowClient().set_registered_model_alias("mock-model", "champion", <new_version_number>)
```

Find the version number either in the training console output, the MLflow UI's Models tab, or:

```python
from mlflow import MlflowClient
client = MlflowClient()
for v in client.search_model_versions("name='mock-model'"):
    print(v.version, v.run_id)
```

Before debugging anything downstream, always confirm which run `champion` actually resolves to:

```python
client.get_model_version_by_alias("mock-model", "champion").run_id
```

## 4. Run inference locally

```
uv run scripts\mock_inference_torch.py
```

Loads `models:/mock-model@champion` and prints predictions for 5 fresh samples. Do this before building
Docker — if it fails here, it'll fail in the container too, and it's much faster to debug outside Docker.

## 5. Build the Docker image

```
docker build --build-arg MLFLOW_TRACKING_URI=http://host.docker.internal:5000 -t mock-model-image .
```

`host.docker.internal` is Docker Desktop's DNS alias for reaching a server running on the host machine — plain
`localhost` from inside a build/container refers to the container itself, not your machine. The build fetches
`pyproject.toml`/`uv.lock`/both scripts/`config/` from whatever run `champion` currently points to (Section 3
above), so it needs the server (Section 1) actually running and reachable at build time.

Optional build args, if training under a different registered model name/alias than the defaults:

```
docker build --build-arg MLFLOW_TRACKING_URI=http://host.docker.internal:5000 --build-arg MODEL_NAME=mock-model --build-arg MODEL_ALIAS=champion -t mock-model-image .
```

## 6. Run the Docker image

```
docker run --rm --gpus all mock-model-image
```

`--rm` removes the container once it exits. `--gpus all` requires the NVIDIA Container Toolkit installed on
the host — sanity check GPU passthrough on its own if inference doesn't seem to be using the GPU:

```
docker run --rm --gpus all mock-model-image python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

Full symptom → cause table: [`lessons/27-mlflow-deployment-pipeline.html`](lessons/27-mlflow-deployment-pipeline.html#cheat-sheet).
Quick reference for the errors most likely to reappear:

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to download artifacts ... please ensure the path is correct` | Client can't reach the artifact store directly | Check the run's `artifact_uri` — must start with `mlflow-artifacts:/`, not `file:...`. If it's `file:...`, `tracking_uri` in `config.yaml` points at a raw file/DB path instead of the server's `http://` address |
| `403 Invalid Host header` | Request's `Host` header isn't allowlisted | Restart the server with `--allowed-hosts` including whatever host you're connecting from |
| Model loads but looks wrong/old | `champion` wasn't reassigned after the last retrain | Section 3 — check `get_model_version_by_alias(...).run_id` directly |
| `Readme file does not exist: README.md` during build | `uv` trying to build the local project's own wheel | Confirm `--no-install-project` is still in the Dockerfile's `uv sync` step |
| `Primary config directory not found` (Hydra) | `config_path="../config"` resolved wrong because the image's directory layout doesn't mirror the local repo | Confirm `scripts/` and `config/` are siblings inside the image, same as locally |
