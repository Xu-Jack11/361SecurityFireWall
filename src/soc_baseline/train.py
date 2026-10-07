"""Command-line training pipeline for the SOC baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from .constants import FEATURE_COLUMNS, ID_COLUMN, LABELS
from .data import detect_label_column, read_parquet_frame, stratified_sample
from .features import FEATURE_SETS, build_log_documents
from .gpu_modeling import TorchTfidfClassifier, TorchTrainingConfig, torch_cuda_available
from .modeling import (
    evaluate_predictions,
    make_model_pipeline,
    write_classification_report,
    write_confusion_matrix,
    write_feature_importance,
    write_label_distribution,
    write_metrics,
)
from .decision import weighted_argmax
from .source_mask import allowed_label_matrix, fit_source_label_mask, source_keys
from .submission import validate_submission_frame


@dataclass(frozen=True)
class BaselineConfig:
    train_path: Path = Path("data/train.parquet")
    test_path: Path = Path("data/valid_input.parquet")
    output_path: Path = Path("res.csv")
    artifacts_dir: Path = Path("artifacts")
    max_train_rows: int | None = 300_000
    max_test_rows: int | None = None
    test_size: float = 0.2
    max_features: int = 120_000
    min_df: int = 3
    top_features: int = 40
    prediction_chunk_size: int = 200_000
    random_state: int = 42
    feature_set: str = "full"
    source_label_mask: bool = False
    source_mask_min_rows: int = 100
    benign_weight: float = 1.0
    model_backend: str = "auto"
    device: str = "auto"
    torch_epochs: int = 3
    torch_batch_size: int = 8192
    torch_learning_rate: float = 0.05
    torch_weight_decay: float = 1e-5


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    """Train, evaluate, refit, predict, and write all baseline artifacts."""

    artifacts_dir = Path(config.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_parquet_frame(config.train_path)
    label_col = detect_label_column(train_df)
    _validate_training_labels(train_df[label_col])
    sampled_train = stratified_sample(train_df, label_col, config.max_train_rows, config.random_state)

    write_label_distribution(
        sampled_train[label_col],
        artifacts_dir / "label_distribution.csv",
        artifacts_dir / "label_distribution.png",
    )

    documents = build_log_documents(sampled_train, config.feature_set)
    labels = sampled_train[label_col].astype(str)
    sources = source_keys(sampled_train)

    train_docs, holdout_docs, train_labels, holdout_labels = _split_train_holdout(
        documents,
        labels,
        config.test_size,
        config.random_state,
    )

    evaluation_model = _make_configured_model(config)
    evaluation_model.fit(train_docs, train_labels)
    # The holdout mask is fit on the fit split only so holdout labels never leak into it.
    evaluation_mask = (
        fit_source_label_mask(sources.loc[train_docs.index], train_labels, config.source_mask_min_rows)
        if config.source_label_mask
        else None
    )
    holdout_predictions = pd.Series(
        _predict(
            evaluation_model,
            holdout_docs,
            sources.loc[holdout_docs.index],
            evaluation_mask,
            config.benign_weight,
        ),
        index=holdout_labels.index,
    )
    probabilities = evaluation_model.predict_proba(holdout_docs) if hasattr(evaluation_model, "predict_proba") else None
    metrics = evaluate_predictions(
        holdout_labels,
        holdout_predictions,
        probabilities,
        _model_classes(evaluation_model),
    )
    model_info = _model_info(evaluation_model)
    metrics.update(
        {
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "label_column": label_col,
            "source_train_rows": int(len(train_df)),
            "sampled_train_rows": int(len(sampled_train)),
            "fit_rows": int(len(train_docs)),
            "holdout_rows": int(len(holdout_docs)),
            "max_features": int(config.max_features),
            "min_df": int(config.min_df),
            "feature_set": config.feature_set,
            "source_label_mask": bool(config.source_label_mask),
            "benign_weight": float(config.benign_weight),
            "model_backend": model_info["model_backend"],
            "device": model_info["device"],
        }
    )
    write_metrics(metrics, artifacts_dir / "metrics.json")
    write_classification_report(holdout_labels, holdout_predictions, artifacts_dir / "classification_report.csv")
    write_confusion_matrix(
        holdout_labels,
        holdout_predictions,
        artifacts_dir / "confusion_matrix.csv",
        artifacts_dir / "confusion_matrix.png",
    )

    final_model = _make_configured_model(config)
    final_model.fit(documents, labels)
    joblib.dump(final_model, artifacts_dir / "model.joblib")
    write_feature_importance(final_model, artifacts_dir / "feature_importance.csv", config.top_features)

    final_mask = None
    if config.source_label_mask:
        final_mask = fit_source_label_mask(
            source_keys(train_df), train_df[label_col], config.source_mask_min_rows
        )
        (artifacts_dir / "source_label_mask.json").write_text(
            json.dumps(final_mask, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    submission_rows = predict_parquet_to_submission(
        final_model,
        config.test_path,
        config.output_path,
        config.prediction_chunk_size,
        config.max_test_rows,
        feature_set=config.feature_set,
        source_mask=final_mask,
        benign_weight=config.benign_weight,
    )
    expected_ids = read_parquet_frame(config.test_path, columns=[ID_COLUMN])[ID_COLUMN]
    if config.max_test_rows is not None:
        expected_ids = expected_ids.head(config.max_test_rows)
    submission = pd.read_csv(config.output_path)
    validate_submission_frame(submission, expected_ids, LABELS)

    metrics["submission_rows"] = int(submission_rows)
    metrics["output_path"] = str(config.output_path)
    write_metrics(metrics, artifacts_dir / "metrics.json")
    return metrics


def predict_parquet_to_submission(
    model: Any,
    test_path: str | Path,
    output_path: str | Path,
    chunk_size: int,
    max_rows: int | None,
    feature_set: str = "full",
    source_mask: dict[str, list[str]] | None = None,
    benign_weight: float = 1.0,
) -> int:
    """Predict a parquet test file in batches and stream a submission CSV.

    ``feature_set`` must match the one the model was trained with.
    """

    path = Path(test_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(path)
    schema_columns = set(parquet.schema.names)
    columns = [ID_COLUMN] + [column for column in FEATURE_COLUMNS if column in schema_columns]

    written = 0
    first = True
    for batch in parquet.iter_batches(batch_size=chunk_size, columns=columns):
        frame = batch.to_pandas()
        if max_rows is not None:
            remaining = max_rows - written
            if remaining <= 0:
                break
            frame = frame.head(remaining)
        if frame.empty:
            continue
        documents = build_log_documents(frame, feature_set)
        predictions = _predict(model, documents, source_keys(frame), source_mask, benign_weight)
        output_frame = pd.DataFrame({ID_COLUMN: frame[ID_COLUMN].astype(str), "pred_label": predictions})
        output_frame.to_csv(output, index=False, mode="w" if first else "a", header=first)
        written += len(output_frame)
        first = False
    return written


def _predict(
    model: Any,
    documents: pd.Series,
    sources: pd.Series,
    source_mask: dict[str, list[str]] | None,
    benign_weight: float = 1.0,
) -> Any:
    if source_mask is None and benign_weight == 1.0:
        return model.predict(documents)
    classes = _model_classes(model)
    allowed = allowed_label_matrix(sources, source_mask, classes) if source_mask is not None else None
    return weighted_argmax(model.predict_proba(documents), classes, allowed, {"benign": benign_weight})


def _split_train_holdout(
    documents: pd.Series,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if test_size <= 0:
        return documents, documents.iloc[:0], labels, labels.iloc[:0]
    stratify = labels if labels.value_counts().min() >= 2 else None
    return train_test_split(
        documents,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _validate_training_labels(labels: pd.Series) -> None:
    unknown = sorted(set(labels.dropna().astype(str)) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown training labels: {unknown}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOC threat-detection baseline and write res.csv")
    parser.add_argument("--train-path", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/valid_input.parquet"))
    parser.add_argument("--output", dest="output_path", type=Path, default=Path("res.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-rows", type=_optional_int, default=300_000)
    parser.add_argument("--max-test-rows", type=_optional_int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=120_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--prediction-chunk-size", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="full",
        help="full: original tokens incl. time and identifiers; content: drops them (see features.py).",
    )
    parser.add_argument(
        "--source-label-mask",
        action="store_true",
        help="Only predict labels each vendor/product source was seen with in training.",
    )
    parser.add_argument(
        "--source-mask-min-rows",
        type=int,
        default=100,
        help="Training rows a source needs before its label set is enforced.",
    )
    parser.add_argument(
        "--benign-weight",
        type=float,
        default=1.0,
        help="Relative cost of missing a benign row; >1 only alerts when p(alert) > weight * p(benign).",
    )
    parser.add_argument(
        "--model-backend",
        choices=("auto", "gpu", "torch", "cpu", "sklearn"),
        default="auto",
        help="auto uses PyTorch CUDA when available, otherwise sklearn CPU; gpu requires CUDA.",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--torch-epochs", type=int, default=3)
    parser.add_argument("--torch-batch-size", type=int, default=8192)
    parser.add_argument("--torch-learning-rate", type=float, default=0.05)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    return int(value)


def _make_configured_model(config: BaselineConfig) -> Any:
    backend = config.model_backend.lower()
    if backend in {"cpu", "sklearn"}:
        return make_model_pipeline(config.max_features, config.min_df, config.random_state)
    if backend == "auto" and not torch_cuda_available():
        return make_model_pipeline(config.max_features, config.min_df, config.random_state)

    require_cuda = backend in {"auto", "gpu"}
    device = "cuda" if backend == "gpu" and config.device == "auto" else config.device
    if backend not in {"auto", "gpu", "torch"}:
        raise ValueError(f"Unknown model backend: {config.model_backend}")

    return TorchTfidfClassifier.from_config(
        TorchTrainingConfig(
            max_features=config.max_features,
            min_df=config.min_df,
            random_state=config.random_state,
            device=device,
            require_cuda=require_cuda,
            epochs=config.torch_epochs,
            batch_size=config.torch_batch_size,
            learning_rate=config.torch_learning_rate,
            weight_decay=config.torch_weight_decay,
        )
    )


def _model_classes(model: Any) -> list[str]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    return list(model.named_steps["classifier"].classes_)


def _model_info(model: Any) -> dict[str, str]:
    if isinstance(model, TorchTfidfClassifier):
        return {"model_backend": "torch", "device": str(model.runtime_device_ or model.device)}
    return {"model_backend": "sklearn", "device": "cpu"}


def main() -> None:
    args = _parse_args()
    config = BaselineConfig(**vars(args))
    result = run_baseline(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
