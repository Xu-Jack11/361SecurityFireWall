"""Model creation, evaluation, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .constants import LABELS


def make_model_pipeline(
    max_features: int,
    min_df: int,
    random_state: int,
) -> Pipeline:
    """Create the baseline TF-IDF + class-balanced linear log-loss pipeline."""

    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=0.98,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[\w.\-/:=@+]+\b",
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    class_weight="balanced",
                    loss="log_loss",
                    max_iter=40,
                    random_state=random_state,
                    tol=1e-3,
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    probabilities: Any | None,
    classes: list[str],
) -> dict[str, Any]:
    """Compute robust holdout metrics for imbalanced multiclass labels."""

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0)
        ),
    }
    if probabilities is not None and len(set(y_true)) > 1:
        try:
            probability_frame = pd.DataFrame(probabilities, columns=classes)
            auc_labels = sorted(label for label in LABELS if label in set(classes))
            if set(y_true.astype(str)) != set(auc_labels):
                raise ValueError("AUC requires all configured labels to be present in y_true")
            probability_frame = probability_frame.reindex(columns=auc_labels, fill_value=0.0)
            row_sums = probability_frame.sum(axis=1).replace(0.0, 1.0)
            probability_frame = probability_frame.div(row_sums, axis=0)
            metrics["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true,
                    probability_frame.to_numpy(),
                    labels=auc_labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["macro_roc_auc_ovr"] = None
    else:
        metrics["macro_roc_auc_ovr"] = None
    return metrics


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_classification_report(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_path)


def write_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, csv_path: Path, png_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    matrix_frame = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
    matrix_frame.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(LABELS)).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Holdout Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def write_label_distribution(labels: pd.Series, csv_path: Path, png_path: Path) -> None:
    counts = labels.value_counts().reindex(LABELS, fill_value=0)
    counts.rename_axis("label").reset_index(name="count").to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_title("Training Label Distribution")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def write_feature_importance(model: Any, output_path: Path, top_n: int) -> None:
    """Export top positive TF-IDF tokens per class from the linear classifier."""

    if hasattr(model, "named_steps"):
        vectorizer = model.named_steps["vectorizer"]
        classifier = model.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        classes = list(classifier.classes_)
        coefficients = classifier.coef_
    else:
        feature_names = model.get_feature_names_out()
        classes = list(model.classes_)
        coefficients = model.coef_

    records: list[dict[str, Any]] = []
    if coefficients.shape[0] == 1 and len(classes) == 2:
        class_coef_pairs = [(classes[1], coefficients[0]), (classes[0], -coefficients[0])]
    else:
        class_coef_pairs = list(zip(classes, coefficients))

    for class_label, coef in class_coef_pairs:
        top_indices = coef.argsort()[::-1][:top_n]
        for rank, feature_idx in enumerate(top_indices, start=1):
            records.append(
                {
                    "class": class_label,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "weight": float(coef[feature_idx]),
                }
            )
    pd.DataFrame(records).to_csv(output_path, index=False)
