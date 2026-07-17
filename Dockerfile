# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Fetches artifacts from MLflow and installs all dependencies. This stage is discarded after
# the build -- none of its layers (bootstrap mlflow, fetch_artifacts.py, uv.lock) end up in
# the final image.
#
# NOTE: the fetch step needs network access to MLFLOW_TRACKING_URI *at build time*. If your
# build environment (CI, etc.) can't reach the tracking server, move the fetch + `uv sync`
# steps into an entrypoint script that runs at container *startup* instead -- slower first
# boot, but works when the build environment is network-isolated from the tracking server.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ARG MODEL_NAME=mock-model
ARG MODEL_ALIAS=champion
ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Minimal bootstrap: just enough to run the fetch script and talk to the tracking server.
# The *real* project dependencies come from the pyproject.toml/uv.lock this step pulls down --
# this mlflow install is intentionally separate from that, chicken-and-egg (we need mlflow to
# fetch the lockfile that pins mlflow's own exact version). --system installs into this
# stage's system Python, never touches the project .venv that gets carried to stage 2.
RUN uv pip install --system --no-cache mlflow

COPY scripts/fetch_artifacts.py .
RUN python fetch_artifacts.py --model-name ${MODEL_NAME} --alias ${MODEL_ALIAS} --dest /app

# /app now has pyproject.toml, uv.lock, scripts/mock_train_torch.py, scripts/mock_inference_torch.py,
# and config/ -- every one of them fetched from the training run, none copied from local disk.
# --no-install-project: this mock example only needs mock_train_torch.py, not the local
# `sequences_to_multiple_outcomes` package pyproject.toml declares as a build target -- and
# that source isn't in this image. For the real project, drop this flag and also fetch
# `src/sequences_to_multiple_outcomes` (logged as an artifact the same way) before syncing.
RUN uv sync --frozen --no-install-project


# ── Stage 2: final ────────────────────────────────────────────────────────────
# Starts fresh from a clean base -- no build tools, no bootstrap mlflow, no fetch script, no
# uv.lock (never needed here since --no-sync means the lockfile is never re-consulted at
# runtime). Only what's needed to actually run inference gets copied forward.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

# Re-declared here, separately from the builder stage -- ARG/ENV don't survive a fresh FROM.
# Without this, MLFLOW_TRACKING_URI would be unset at container *runtime*, silently falling
# back to whatever tracking_uri happens to be baked into the fetched config.yaml instead.
ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# the installed venv (built in stage 1 against the exact fetched uv.lock) + app files -- same
# base image and WORKDIR in both stages, so paths inside .venv resolve correctly after the copy
COPY --from=builder /app/.venv      ./.venv
COPY --from=builder /app/scripts    ./scripts
COPY --from=builder /app/config     ./config
COPY --from=builder /app/pyproject.toml .

# --no-sync: the venv was already built correctly in stage 1 (with --no-install-project) --
# `uv run` would otherwise re-sync on every container start, which tries to build the local
# `a-study-on-transformer` project itself (missing README.md/src/ in this image) and fails.
# pyproject.toml still needs to be present for uv to recognize the project root and locate
# .venv -- only uv.lock itself is genuinely unneeded here, since sync never runs in this stage.
CMD ["uv", "run", "--no-sync", "scripts/mock_inference_torch.py"]
