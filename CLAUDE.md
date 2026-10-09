# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible baseline for a SOC (Security Operations Center) security-event
classification challenge. It trains a class-balanced TF-IDF linear model over
security log text plus structured field tokens, evaluates a holdout split,
writes diagnostics, and streams predictions in the submission format
(`event_id,pred_label`). Labels are 3-class: `benign`, `suspicious`,
`malicious` (see `LABELS` in `src/soc_baseline/constants.py`).

The `data/` directory (git-ignored) contains the parquet inputs:
`data/train.parquet` (labeled, with a `label_binary` column),
`data/valid_input.parquet` and `data/valid_answer_private.parquet`.

**`valid_input` is the validation set, not the test set.** Its labels are in
`valid_answer_private.parquet`. Use them only to score predictions
(`scripts/score_submissions.py`, `soc_hybrid.evaluate`), never for training or
tuning. The competition test set is not in `data/`. `res.csv` is the name of
the competition submission, so write it only when predicting that test set
(`--test-path <test parquet> --output res.csv`). Don't produce a new `res.csv`
from valid. Prediction runs on valid write to their artifacts dir
(`predictions.csv` / `valid_pred.csv`). Older files named `res*.csv` (the root
`res.csv` and the files under `artifacts/`) are historical predictions on valid,
kept because the scoring scripts and the iteration log reference them. Iteration log
entries before I11 call valid the "测试集".

Scoring (corrected 2026-10-08): the competition scores threat detection and the 3-class
split together. A threat (suspicious or malicious) predicted as benign is penalized most;
suspicious and malicious predicted as each other earn partial credit. The exact weights are
not known, so `soc_hybrid` costs every result with `decision.cost_matrix(m)` at m = 2, 5, 10.
Benign-first tuning (`--benign-weight`, I12, and I16's `final4`) targets the opposite error.

## Environment setup

`uv.toml` pins uv to the Zhejiang University PyPI mirror. Create the venv with:

```bash
bash scripts/setup_uv_env.sh   # uses --system-site-packages to reuse the runner's CUDA PyTorch build
```

On a fresh machine without a preinstalled PyTorch, use the GPU extra instead:
`uv venv .venv && uv sync --extra gpu`. PyTorch is an **optional** dependency;
the code imports it lazily (`gpu_modeling._import_torch`) so the sklearn path
works with no torch installed.

## Common commands

Run the full pipeline (train → evaluate → refit → predict → write artifacts):

```bash
python -m soc_baseline.train \
  --train-path data/train.parquet --test-path data/valid_input.parquet \
  --artifacts-dir artifacts \
  --max-train-rows 300000 --model-backend gpu --device cuda
```

Without `--output`, predictions go to `<artifacts-dir>/predictions.csv`.

Time-free variant (I13 in `docs/iteration_log.md`, the recommended one): add
`--feature-set timefree --source-label-mask`, then run
`python scripts/apply_verdict_rule.py --submission <pred.csv> --output <out.csv>`
(add `--benign-weight 10` to training when missed benign rows are the costlier
error, I12). `scripts/check_time_independence.py --verdict-rule` verifies that
rewriting every timestamp, digit and calendar word changes no prediction, and
`scripts/eval_external.py` scores all versions on samples of the external
WitFoo dumps in `data/external/` (I14: on par with the date rule on v4/latest,
no malicious recall on v2, whose labels no single record can recover).
`--feature-set content` (I9) still leaks the year through half-sanitized
epochs, and `scripts/apply_date_rule.py` is time-based; both are kept for
comparison. The default `--feature-set full` is kept only to reproduce the
original baseline — its time tokens leak the label and drive ~44% malicious
predictions on valid.

Predict from saved weights without retraining (an artifacts dir or the
released `weights/timefree`): `python scripts/predict.py --model-dir DIR
--output OUT.csv [--test-path P] [--verdict-rule weights/verdict_rule.json]`;
`--output` is required, `--test-path` defaults to valid, and it reads the
feature set and benign weight from `DIR/metrics.json`.

LLM routing (I15): `scripts/llm_reasoning_probe.py` runs a local LLM (vLLM,
Qwen3.5-4B from ModelScope at `/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B`)
zero-shot over one event at a time. `sample`/`run`/`score` evaluate it on a
probe sample; `route --no-thinking` keeps G and sends only rows of sources with
< 100 train rows to the LLM, writing `artifacts/llm_probe/Qwen3.5-4B-nothink/res_route.csv`
(99 errors on valid vs 864 for G). Use `.venv/bin/python`; the T4 memory
settings in `run()` are needed to avoid OOM, and thinking mode costs ~17× the
tokens for no accuracy gain.

Classifier + LLM hybrid (I16, I17): `src/soc_hybrid/` is a separate package built from
train only (it uses no `soc_baseline` model, rule or prediction). Stages:
`audit` → `holdout` (ud5/cl5/LOSO/LOCO) → `novelty` → `llm_holdout` →
`cost_select` (corrected rule, I17; `route_select` / `classifier_select` did the
benign-first selection of I16) → `pipeline` (predicts valid, writes
`artifacts/hybrid/runs/<name>/valid_pred.csv`; runs made before this rename
have `res.csv`) → `evaluate` / `ablation` (the only steps that read valid
labels). The current config is the `cost2` command in the iteration log (I18;
`cost1` in I17 is the same with a 2,000-document LLM budget): `--miss-weight`
switches the classifier to the minimum-cost decision, `--fusion gate_km|raise|mix_km`
to the corrected-rule fusions, and `--budget 0` sends every triggered document to the LLM.
`python -m soc_hybrid.external` (I19) runs that config unchanged on 10,000 random rows of each
`data/external` group and scores them there; `python -m soc_hybrid.v2_failure` (I20) explains why v2's malicious rows
come out suspicious (v2 labels incident links; the competition's malicious rows are the same events exported without
vendor fields, which is what the classifier keys on), and `python -m soc_hybrid.vendor_blind` (I21) refits the
classifier without vendor fields (scored on valid by `soc_hybrid.evaluate --runs vendor_blind`). LLM replies are
cached in `artifacts/hybrid/llm/*.jsonl`.
vLLM must be started from a script or module, not stdin (it spawns workers), and
`soc_hybrid.llm` pins `VLLM_PORT` above the ephemeral range because proxy
connections can hold vLLM's default port.

Fast smoke run: `python -m soc_baseline.train --max-train-rows 50000 --max-test-rows 200000`
Force CPU: add `--model-backend sklearn`.

Tests (unittest, no pytest):

```bash
python -m unittest discover -s tests -v          # all
python -m unittest tests.test_features -v        # one module
python -m unittest tests.test_features.TestClass.test_name   # one test
```

Note `tests/test_gpu_modeling.py` skips itself when torch/CUDA is unavailable.

## Architecture

The package lives in `src/soc_baseline/` (src-layout; installed as
`soc-threat-baseline`). Data flows in one direction through these modules:

- **`constants.py`** — single source of truth for column names and label set.
  `FEATURE_COLUMNS`/`CATEGORICAL_COLUMNS` drive feature building; `LABELS`
  fixes both label order and the allowed submission values. Change schema
  assumptions here, not in downstream modules.
- **`data.py`** — parquet loading, label-column autodetection
  (`LABEL_CANDIDATES`), and `stratified_sample`, which allocates per-class row
  quotas proportionally (largest-remainder method) so every class survives
  downsampling even under `--max-train-rows`.
- **`features.py`** — `build_log_documents` turns a DataFrame into one text
  "document" per event by concatenating `field=value` tokens: cleaned
  categoricals, a `src_port` bucket, IPv4 `/24` prefixes, hour/dow/month from
  the unix `timestamp`, and the raw `message_sanitized` text. This string is
  the model's entire input surface — both training and prediction go through
  it, so feature changes here affect both. `feature_set="content"` instead
  keeps only pipeline/product/vendor, field *shapes* (`src_ip_kind=ipv4`), the
  port bucket and the message after `normalize_message` folds dates, times,
  IPs, sanitizer IDs (`USER-0010-0324` → `tok_user`) and long numbers.
  `feature_set="timefree"` goes further: `normalize_message_timefree` maps every
  sanitizer token and digit run to `0` and masks calendar words, so no number
  survives, and `firewall_action` adds a vendor-agnostic `fw_action=block|allow|none`
  token plus its combination with vendor presence. A model must be predicted
  with the same `feature_set` it was trained with; scripts that load
  `model.joblib` default to `full`.
- **`modeling.py`** — the CPU/sklearn path: `make_model_pipeline`
  (TfidfVectorizer 1–2 grams + `class_weight="balanced"` `SGDClassifier` with
  `log_loss`) plus all the diagnostic writers (metrics, classification report,
  confusion matrix, label distribution, feature importance).
- **`gpu_modeling.py`** — the GPU/torch path: `TorchTfidfClassifier` reuses the
  same TfidfVectorizer but trains a linear head with AdamW + class-weighted
  cross-entropy over CSR→sparse-COO minibatches. It is **duck-typed to match
  the sklearn pipeline's interface** (`fit`/`predict`/`predict_proba`/`coef_`/
  `get_feature_names_out`/`classes_`) so the rest of the pipeline treats both
  backends identically. `__getstate__` moves tensors to CPU before pickling so
  `joblib.dump` produces a portable model.
- **`source_mask.py`** — optional decision rule: for each `vendor/product`
  source with ≥ `--source-mask-min-rows` training rows, only labels that
  source was seen with are allowed (argmax of `predict_proba` over the
  allowed set); rare or unseen sources stay unrestricted. The mask is saved
  as `source_label_mask.json`.
- **`decision.py`** — `weighted_argmax`: argmax of `p(y|x) * weight[y]` over
  the allowed labels. `--benign-weight w` only raises suspicious/malicious when
  it is > w× as likely as benign. `train._predict` routes through it whenever
  a mask or a non-1 weight is set; otherwise it calls `model.predict`.
- **`verdict_rule.py`** — post-prediction rule used by
  `scripts/apply_verdict_rule.py`: learns from the training labels that the
  vendor-less + block-verdict cell is malicious (support and purity checked,
  raises otherwise) and forces that label on matching predicted rows. It is the
  time-free replacement for the date rule and, like it, exploits how the
  dataset was built rather than a transferable security signal.
- **`submission.py`** — `validate_submission_frame` enforces the exact output
  contract (columns, no duplicate/missing/extra `event_id`, only allowed
  labels, row-count match). The pipeline validates its own prediction CSV
  before finishing.
- **`train.py`** — orchestrator + CLI. `BaselineConfig` holds all knobs;
  `_make_configured_model` resolves the `--model-backend`
  (`auto`/`gpu`/`torch`/`cpu`/`sklearn`) into a concrete estimator (`auto`
  falls back to sklearn when CUDA is absent). It trains a holdout model for
  metrics, then **refits a fresh model on the full sample** for prediction,
  and streams predictions over the `--test-path` parquet (valid by default) in
  batches (`predict_parquet_to_submission`) to bound memory on the large input.

### Backend abstraction — the key invariant

Both estimators expose the same interface, and `train.py` only ever calls that
interface. When adding features or metrics, keep this parity: anything the
sklearn pipeline exposes (e.g. `coef_`, `predict_proba`) must also work on
`TorchTfidfClassifier`, or the diagnostic writers in `modeling.py` will break
for one backend.

## Design docs

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold the original
design spec and implementation plan. `docs/public_datasets.md` surveys external
public cybersecurity datasets (intentionally **not** merged into training data —
schemas and label definitions differ from the competition).
