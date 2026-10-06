# SOC Threat Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible SOC threat-classification baseline that outputs a valid `res.csv` and documents public cybersecurity datasets.

**Architecture:** A small Python package reads parquet data, converts each event to a text document made from structured field tokens and sanitized messages, trains a class-balanced TF-IDF linear log-loss model with PyTorch CUDA when requested, exports diagnostics, and writes chunked predictions. A sklearn CPU backend remains available for fallback. Documentation records dataset findings and usage guidance.

**Tech Stack:** Python 3.10+, uv, pandas, pyarrow, scikit-learn, PyTorch, joblib, matplotlib, unittest.

## Global Constraints

- Do not commit `data/`, `artifacts/`, model files, or `res.csv`.
- Submission labels must be exactly `benign`, `suspicious`, or `malicious`.
- Submission rows must match `valid_input.parquet` `event_id` values exactly.
- The baseline must work when `message_sanitized` is blank.
- GPU runs must record `model_backend=torch` and `device=cuda` in
  `artifacts/metrics.json`.
- uv must use `https://mirrors.zju.edu.cn/pypi/web/simple` as the configured
  package source.
- Tests must run with `python -m unittest discover -s tests -v`.

---

### Task 1: Repository Skeleton And Tests

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `docs/superpowers/specs/2026-07-09-soc-threat-baseline-design.md`
- Create: `docs/superpowers/plans/2026-07-09-soc-threat-baseline.md`
- Create: `tests/test_features.py`
- Create: `tests/test_data.py`
- Create: `tests/test_submission.py`
- Create: `tests/test_train_smoke.py`
- Create: `tests/test_gpu_modeling.py`
- Create: `tests/test_uv_config.py`

**Interfaces:**
- Produces expected APIs for later tasks:
  - `soc_baseline.features.build_log_documents(df: pandas.DataFrame) -> pandas.Series`
  - `soc_baseline.features.port_bucket(value: object) -> str`
  - `soc_baseline.data.detect_label_column(df: pandas.DataFrame) -> str`
  - `soc_baseline.data.stratified_sample(df: pandas.DataFrame, label_col: str, max_rows: int | None, random_state: int) -> pandas.DataFrame`
  - `soc_baseline.submission.validate_submission_frame(submission: pandas.DataFrame, expected_event_ids: pandas.Series, allowed_labels: tuple[str, ...]) -> None`
  - `soc_baseline.submission.write_submission(event_ids, labels, output_path) -> pathlib.Path`
  - `soc_baseline.train.run_baseline(config: BaselineConfig) -> dict`
  - `soc_baseline.gpu_modeling.TorchTfidfClassifier`

- [x] **Step 1: Write failing tests**
- [ ] **Step 2: Run tests and confirm imports fail because package code does not exist**

### Task 2: Core Data And Feature Modules

**Files:**
- Create: `src/soc_baseline/__init__.py`
- Create: `src/soc_baseline/constants.py`
- Create: `src/soc_baseline/features.py`
- Create: `src/soc_baseline/data.py`

**Interfaces:**
- Consumes: tests from Task 1.
- Produces: reusable feature documents and data utilities.

- [ ] **Step 1: Implement constants and document builder**
- [ ] **Step 2: Implement parquet label detection and stratified sampling**
- [ ] **Step 3: Run focused tests**

### Task 3: Modeling, Submission, And Training CLI

**Files:**
- Create: `src/soc_baseline/modeling.py`
- Create: `src/soc_baseline/gpu_modeling.py`
- Create: `src/soc_baseline/submission.py`
- Create: `src/soc_baseline/train.py`
- Create: `uv.toml`
- Create: `scripts/setup_uv_env.sh`

**Interfaces:**
- Consumes: `build_log_documents`, `stratified_sample`, and submission validators.
- Produces: `BaselineConfig`, `run_baseline`, `python -m soc_baseline.train`,
  `--model-backend gpu`, `--device cuda`, and uv environment setup.

- [ ] **Step 1: Implement scikit-learn pipeline and metrics export**
- [ ] **Step 2: Implement chunked test prediction and strict submission writing**
- [ ] **Step 3: Run all unit tests**

### Task 4: Public Dataset Research

**Files:**
- Create: `docs/public_datasets.md`

**Interfaces:**
- Consumes: official public web sources.
- Produces: a ranked list of external datasets and practical usage guidance.

- [ ] **Step 1: Document datasets from UNB CIC, UNSW, LANL, CMU SEI, Splunk, OTRF, and DARPA OpTC**
- [ ] **Step 2: Include source links, relevance, label/schema fit, and cautions**

### Task 5: Verification Run

**Files:**
- Modify/generated ignored outputs: `artifacts/*`, `res.csv`

**Interfaces:**
- Consumes: the implemented package and local parquet data.
- Produces: verified tests, metrics artifact, and valid submission file.

- [ ] **Step 1: Run `python -m unittest discover -s tests -v`**
- [ ] **Step 2: Run a smoke baseline with a small train/test limit**
- [ ] **Step 3: Run the baseline command that produces `res.csv`**
- [ ] **Step 4: Verify `res.csv` row count, event-id coverage, label values, and duplicate status**
