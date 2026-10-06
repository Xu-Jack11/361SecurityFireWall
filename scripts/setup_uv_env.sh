#!/usr/bin/env bash
set -euo pipefail

uv venv --python /usr/bin/python3.12 --system-site-packages .venv
uv pip install -e .

.venv/bin/python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
