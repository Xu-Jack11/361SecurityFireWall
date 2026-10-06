# SOC Threat Detection Baseline

This repository contains a reproducible baseline for the SOC security event
classification challenge. The local `data/` directory is expected to contain:

- `data/train.parquet`: labeled training logs with `label_binary`
- `data/valid_input.parquet`: unlabeled test/validation input

The baseline trains a class-balanced TF-IDF model on log text plus structured
field tokens, evaluates a holdout split, writes model diagnostics, and
generates the required `res.csv` submission. When CUDA PyTorch is available,
the default `auto` backend trains the linear classifier on GPU.

## Setup

`uv.toml` configures uv to use the Zhejiang University PyPI mirror through
`index-url`:
`https://mirrors.zju.edu.cn/pypi/web/simple`.

```bash
bash scripts/setup_uv_env.sh
```

The script creates `.venv` with uv and installs the project. It uses
`--system-site-packages` so this runner can reuse the already installed CUDA
PyTorch build. On a fresh machine without PyTorch, install the optional GPU
extra instead:

```bash
uv venv .venv
uv sync --extra gpu
```

## Run Baseline

```bash
python -m soc_baseline.train \
  --train-path data/train.parquet \
  --test-path data/valid_input.parquet \
  --output res.csv \
  --artifacts-dir artifacts \
  --max-train-rows 300000 \
  --model-backend gpu \
  --device cuda
```

Use `--model-backend sklearn` for the original CPU fallback.

The default feature set reproduces the original baseline, whose timestamp and
identifier tokens leak the label (see `docs/iteration_log.md`, I9). For the
leak-free model with the per-source label mask add:

```bash
  --feature-set content --source-label-mask
```

Generated files:

- `res.csv`: required submission with `event_id,pred_label`
- `artifacts/model.joblib`: trained final model
- `artifacts/metrics.json`: holdout metrics
- `artifacts/classification_report.csv`: per-class precision/recall/F1
- `artifacts/confusion_matrix.csv` and `.png`
- `artifacts/feature_importance.csv`: top weighted tokens per class

For a fast smoke run:

```bash
python -m soc_baseline.train --max-train-rows 50000 --max-test-rows 200000
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Public Dataset Research

Additional authoritative public cybersecurity datasets are summarized in
[`docs/public_datasets.md`](docs/public_datasets.md). They are not merged into
the challenge training data by default because external labels and feature
schemas differ from the competition schema; they are best used for pretraining,
robustness checks, feature ideas, and attack-chain analysis experiments.
