"""Restrict predictions to the labels each log source produced in training.

In the training data the label set is largely a property of the log source
(e.g. AWS Instance Backup is 100% benign, Cisco ASA is benign/suspicious and
never malicious). For a well-supported source we therefore never predict a
label it has not been seen with; sources that are rare or unseen in training
are left unrestricted so the model decides on its own.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .decision import weighted_argmax


def source_keys(df: pd.DataFrame) -> pd.Series:
    """Identify each row's log source as ``vendor_name/product_name``."""

    def column(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str).str.strip()
        return pd.Series("", index=df.index, dtype="object")

    return column("vendor_name") + "/" + column("product_name")


def fit_source_label_mask(
    sources: pd.Series,
    labels: pd.Series,
    min_rows: int,
) -> dict[str, list[str]]:
    """Map each source with at least ``min_rows`` rows to its observed labels."""

    counts = pd.crosstab(sources.to_numpy(), labels.astype(str).to_numpy())
    supported = counts[counts.sum(axis=1) >= min_rows]
    return {
        str(source): sorted(str(label) for label in row.index[row > 0])
        for source, row in supported.iterrows()
    }


def predict_with_source_mask(
    model: Any,
    documents: pd.Series,
    sources: pd.Series,
    mask: Mapping[str, list[str]],
    classes: list[str],
) -> np.ndarray:
    """Argmax of ``predict_proba`` over the labels allowed for each row's source."""

    allowed = allowed_label_matrix(sources, mask, classes)
    return weighted_argmax(model.predict_proba(documents), classes, allowed)


def allowed_label_matrix(
    sources: pd.Series,
    mask: Mapping[str, list[str]],
    classes: list[str],
) -> np.ndarray:
    """Boolean (rows x classes) matrix; unrestricted sources allow every class."""

    codes, uniques = pd.factorize(sources.to_numpy())
    per_source = np.ones((len(uniques), len(classes)), dtype=bool)
    for index, source in enumerate(uniques):
        if source in mask:
            allowed = set(mask[source])
            per_source[index] = [label in allowed for label in classes]
    return per_source[codes]
