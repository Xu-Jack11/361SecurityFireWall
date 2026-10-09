"""Metrics for both scoring rules.

The original rule put benign misses first (a benign event raised as an alert cost 10). The corrected rule puts
threat misses first: a suspicious or malicious event called benign costs ``m``, a false alarm 1 and a
suspicious/malicious swap 0.5 (partial credit). ``m`` is not known, so every result is costed at m = 2, 5, 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABELS
from .decision import SWAP_COST, cost_matrix

BENIGN_MISS_COST = 10.0
MISS_COSTS = (2.0, 5.0, 10.0)


def evaluate(truth, pred, weight=None) -> dict:
    """accuracy, macro-F1, per-class P/R/F1, confusion matrix, benign misses and cost.

    ``weight`` counts each item (e.g. rows per unique document); default 1.
    """

    truth = np.asarray(truth, dtype=object)
    pred = np.asarray(pred, dtype=object)
    weight = np.ones(len(truth)) if weight is None else np.asarray(weight, dtype=float)
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = np.zeros((3, 3))
    np.add.at(matrix, (np.array([index[t] for t in truth], dtype=int), np.array([index[p] for p in pred], dtype=int)), weight)
    total = matrix.sum()
    per_class = {}
    f1s = []
    for i, label in enumerate(LABELS):
        support = matrix[i].sum()
        predicted = matrix[:, i].sum()
        precision = matrix[i, i] / predicted if predicted else 0.0
        recall = matrix[i, i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    benign_misses = matrix[0, 1:].sum()
    errors = total - np.trace(matrix)
    threat_misses = matrix[1:, 0].sum()
    swaps = matrix[1, 2] + matrix[2, 1]
    caught = matrix[1:, 1:].sum()
    threat_recall = caught / (caught + threat_misses) if caught + threat_misses else 0.0
    threat_precision = caught / (caught + benign_misses) if caught + benign_misses else 0.0
    return {
        "items": total,
        "accuracy": np.trace(matrix) / total if total else 0.0,
        # Macro over the labels present in the truth, so a holdout without a class is not scored on it.
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "benign_recall": per_class["benign"]["recall"],
        "benign_misses": benign_misses,
        "errors": errors,
        "cost": BENIGN_MISS_COST * benign_misses + (errors - benign_misses),
        # Corrected rule: threat misses first, swaps at partial credit.
        "threat_misses": threat_misses,
        "false_alarms": benign_misses,
        "swaps": swaps,
        "threat_recall": threat_recall,
        "threat_precision": threat_precision,
        "threat_f1": (2 * threat_precision * threat_recall / (threat_precision + threat_recall)
                      if threat_precision + threat_recall else 0.0),
        "partial_accuracy": (np.trace(matrix) + (1 - SWAP_COST) * swaps) / total if total else 0.0,
        **{f"cost_m{m:g}": float((cost_matrix(m) * matrix).sum()) for m in MISS_COSTS},
        "per_class": per_class,
        "confusion": matrix.tolist(),
    }


def summary_row(name: str, result: dict) -> dict:
    return {
        "variant": name,
        "accuracy": round(result["accuracy"], 6),
        "macro_f1": round(result["macro_f1"], 5),
        "benign_recall": round(result["benign_recall"], 6),
        "benign_misses": int(round(result["benign_misses"])),
        "errors": int(round(result["errors"])),
        "cost": int(round(result["cost"])),
        "threat_misses": int(round(result["threat_misses"])),
        "swaps": int(round(result["swaps"])),
        "threat_recall": round(result["threat_recall"], 6),
        **{f"cost_m{m:g}": round(result[f"cost_m{m:g}"], 1) for m in MISS_COSTS},
    }


def confusion_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame(np.rint(result["confusion"]).astype(int), index=[f"true_{l}" for l in LABELS],
                        columns=[f"pred_{l}" for l in LABELS])
