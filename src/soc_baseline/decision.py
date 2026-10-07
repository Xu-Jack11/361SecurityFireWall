"""Turn class probabilities into labels under per-label costs and constraints."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def weighted_argmax(
    probabilities: np.ndarray,
    classes: list[str],
    allowed: np.ndarray | None = None,
    label_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Bayes decision: argmax of ``p(y|x) * weight[y]`` over the allowed labels.

    A weight is the relative cost of missing that label: ``{"benign": 10}``
    only raises a non-benign label when it is >10x as likely as benign.
    Rows whose allowed set is empty fall back to every label.
    """

    scores = np.asarray(probabilities, dtype=np.float64).copy()
    if label_weights:
        for label, weight in label_weights.items():
            if label in classes:
                scores[:, classes.index(label)] *= weight
    if allowed is not None:
        allowed = allowed.copy()
        allowed[~allowed.any(axis=1)] = True
        scores = np.where(allowed, scores, -np.inf)
    return np.asarray(classes)[scores.argmax(axis=1)]
