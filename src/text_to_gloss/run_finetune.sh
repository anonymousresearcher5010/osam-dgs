#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Configurable paths (workspace for code, PVC for data+outputs)
DATASET_CSV="${DATASET_CSV:-/storage/OSAM-DGS_distilled_sentence_gloss_data_bt.csv}"
WORK_ROOT="${WORK_ROOT:-/storage/text_2_gloss_runs}"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
EPOCHS="${EPOCHS:-6}"
SEED="${SEED:-101}"

# Debug control: cap training/eval samples (set -1 for full run)
MAX_STEPS="${MAX_STEPS:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"

# Speed-oriented defaults
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

# Python / venv configuration (uv-managed Python, pinned)
PYTHON_VERSION="${PYTHON_VERSION:-3.12.2}"
VENV_DIR="${VENV_DIR:-/workspace/project/.venv-text2gloss}"
PYTHON_BIN="${VENV_DIR}/bin/python"

# Cache dirs (PVC-backed by default)
export HF_HOME="${HF_HOME:-/storage/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/storage/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/storage/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-/storage/.cache/torch}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# Optional: hide noisy third-party Syntax/Future warnings in logs.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::SyntaxWarning,ignore::FutureWarning}"

mkdir -p "${WORK_ROOT}" "${PROJECT_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${TORCH_HOME}"

cd "${PROJECT_DIR}"

# Ensure system build tools exist (required by triton/bitsandbytes paths)
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates build-essential
rm -rf /var/lib/apt/lists/*

# Ensure uv is present
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ln -sf /root/.local/bin/uv /usr/local/bin/uv
fi

uv --version

# Install/use pinned Python version with uv
uv python install "${PYTHON_VERSION}"

# Create venv with pinned Python if missing
if [ ! -x "${PYTHON_BIN}" ]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

"${PYTHON_BIN}" --version

# Install dependencies into venv (not system Python)
uv pip install --python "${PYTHON_BIN}" --upgrade pip
uv pip install --python "${PYTHON_BIN}" setuptools==81.0.0 wheel==0.47.0
uv pip install --python "${PYTHON_BIN}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uv pip install --python "${PYTHON_BIN}" -r requirements-deepseek.txt

# Persist runtime environment snapshot (for reproducibility/debugging)
STAMP="$(date +%F-%H%M%S)"
ENV_DIR="${WORK_ROOT}/env_snapshots/${STAMP}"
mkdir -p "${ENV_DIR}"

uv pip freeze --python "${PYTHON_BIN}" > "${ENV_DIR}/uv-pip-freeze.txt"
"${PYTHON_BIN}" -m pip list --format=freeze > "${ENV_DIR}/pip-list-freeze.txt"
"${PYTHON_BIN}" - <<'PY' > "${ENV_DIR}/runtime-info.txt"
import os, platform, sys
print(f"python={sys.version}")
print(f"platform={platform.platform()}")
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"gpu_name={torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch_info_error={e}")
for k in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "PIP_CACHE_DIR", "UV_CACHE_DIR", "TORCH_HOME"]:
    print(f"{k}={os.getenv(k)}")
PY

SPLIT_DATASET_NAME="$(basename "${DATASET_CSV}")"
SPLIT_DATASET_STEM="${SPLIT_DATASET_NAME%.*}"
SPLIT_DIR="${WORK_ROOT}/splits/${SPLIT_DATASET_STEM}/seed_${SEED}"
mkdir -p "${SPLIT_DIR}"

TRAIN_CSV="${SPLIT_DIR}/${SPLIT_DATASET_STEM}_train.csv"
DEV_CSV="${SPLIT_DIR}/${SPLIT_DATASET_STEM}_dev.csv"
TEST_CSV="${SPLIT_DIR}/${SPLIT_DATASET_STEM}_test.csv"

"${PYTHON_BIN}" split_dataset.py \
  --input_csv "${DATASET_CSV}" \
  --output_dir "${SPLIT_DIR}" \
  --seed "${SEED}" \
  --output_prefix "${SPLIT_DATASET_STEM}"

"${PYTHON_BIN}" train_text2gloss_deepseek.py \
  --train_csv "${TRAIN_CSV}" \
  --dev_csv "${DEV_CSV}" \
  --test_csv "${TEST_CSV}" \
  --num_epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --run_root "${WORK_ROOT}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"