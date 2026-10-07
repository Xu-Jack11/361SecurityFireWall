"""Label vendor-less firewall "block" records from what training says about them.

In train.parquet a record that arrives without vendor/product metadata and
carries a block verdict (ASA "Deny", Meraki "decision=blocked", WAF
"act=DENY") is malicious in 79,968 of 79,971 rows; the same verdict with a
vendor attached is suspicious. The rule is learned and purity-checked on the
training labels and overrides the model for that one cell. It uses no time
field: it is the content-based replacement for scripts/apply_date_rule.py.

Like the date rule it rests on how the dataset was built (the historical
attack records were exported without vendor metadata), so it is a
competition-specific rule, not a security signal that transfers to a SOC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import firewall_action

RULE_CELL = ("missing", "block")


def verdict_cells(frame: pd.DataFrame) -> pd.Series:
    """Each row's (vendor state, firewall verdict) cell, e.g. ``missing/block``."""

    vendor = frame["vendor_name"] if "vendor_name" in frame.columns else pd.Series("", index=frame.index)
    state = (vendor.fillna("").astype(str).str.strip() == "").map({True: "missing", False: "present"})
    messages = frame["message_sanitized"] if "message_sanitized" in frame.columns else pd.Series("", index=frame.index)
    return state + "/" + firewall_action(messages)


def fit_verdict_rule(
    frame: pd.DataFrame,
    labels: pd.Series,
    min_rows: int = 1000,
    min_purity: float = 0.999,
) -> dict[str, Any]:
    """Return the majority label of the vendor-less block cell, or raise if it is not pure."""

    cell = "/".join(RULE_CELL)
    in_cell = (verdict_cells(frame) == cell).to_numpy()
    counts = labels[in_cell].astype(str).value_counts()
    support = int(counts.sum())
    if support < min_rows:
        raise ValueError(f"{cell} has {support} training rows (< {min_rows}); rule not supported")
    purity = float(counts.iloc[0] / support)
    if purity < min_purity:
        raise ValueError(f"{cell} majority label covers {purity:.5f} (< {min_purity}); rule not pure")
    return {
        "cell": cell,
        "label": str(counts.index[0]),
        "support": support,
        "purity": purity,
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }


def apply_verdict_rule(frame: pd.DataFrame, predictions: Any, rule: dict[str, Any]) -> np.ndarray:
    """Set predictions in the rule's cell to the rule's label."""

    result = np.asarray(predictions, dtype=object).copy()
    result[(verdict_cells(frame) == rule["cell"]).to_numpy()] = rule["label"]
    return result
