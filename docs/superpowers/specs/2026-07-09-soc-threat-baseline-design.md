# SOC Threat Baseline Design

## Goal

Build a reproducible baseline for the competition data already present in
`data/`, produce a valid `res.csv`, and document authoritative public datasets
that can support later SOC detection experiments.

## Current Data Contract

The repository data inspected on 2026-07-09 contains:

- `data/train.parquet`: 2,056,871 rows and 13 columns.
- `data/valid_input.parquet`: 2,014,052 rows and 12 columns.
- Training label column: `label_binary`.
- Label values: `benign`, `suspicious`, `malicious`.
- Required submission columns: `event_id`, `pred_label`.

The actual local schema is narrower than the problem statement. It includes
`event_id`, `timestamp`, `pipeline`, `src_ip`, `dst_ip`, `src_port`,
`src_host`, `dst_host`, `username`, `message_sanitized`, `product_name`, and
`vendor_name`. The baseline must adapt to this schema rather than assume
missing fields such as `severity`, `action`, or `protocol`.

## Recommended Approach

Use a lightweight supervised machine-learning baseline:

- Convert structured fields into stable text tokens, including missing-value
  markers, timestamp buckets, and source-port buckets.
- Append the sanitized raw message when present.
- Vectorize with `TfidfVectorizer`.
- Use PyTorch to train a sparse TF-IDF linear classifier on CUDA when
  `--model-backend auto` detects a GPU or when `--model-backend gpu` is
  requested. Keep the sklearn `SGDClassifier(loss="log_loss")` implementation
  as an explicit CPU fallback via `--model-backend sklearn`.
- Evaluate on a stratified holdout split from a configurable training sample.
- Refit on the sampled training data and predict the full test input in chunks.

This approach is intentionally simple, fast to iterate, and easy to explain.
It is more suitable for a first baseline than a transformer model because the
data is large, messages can be long, and many test messages are blank.

## Components

- `soc_baseline.features`: converts event rows into model documents.
- `soc_baseline.data`: loads parquet files, detects labels, and samples data.
- `soc_baseline.gpu_modeling`: trains and predicts with a PyTorch sparse
  TF-IDF linear classifier on CPU or CUDA.
- `soc_baseline.modeling`: creates the scikit-learn pipeline and diagnostics.
- `soc_baseline.submission`: validates and writes competition submissions.
- `soc_baseline.train`: command-line orchestration for train, evaluate,
  predict, and artifact export.
- `uv.toml` and `scripts/setup_uv_env.sh`: create the uv environment using the
  Zhejiang University PyPI mirror and Python 3.12 so CUDA PyTorch is available.

## Outputs

- `res.csv` must contain exactly one row for each `event_id` in
  `valid_input.parquet`.
- Predictions must be only `benign`, `suspicious`, or `malicious`.
- Diagnostic artifacts go under `artifacts/` and are ignored by git.
- Public dataset research is saved in `docs/public_datasets.md`.

## Testing

Use standard-library `unittest` so tests run without adding a test dependency.
Tests cover field-to-document conversion, stratified sampling, submission
validation, and a small end-to-end train/predict smoke path.
