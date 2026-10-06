"""Diagnose whether the SOC baseline's near-perfect holdout is due to label
leakage / trivially separable fields, and whether it will generalize to the
valid_input test set.

Run: .venv/bin/python scripts/check_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = Path("data/train.parquet")
VALID = Path("data/valid_input.parquet")
LABEL = "label_binary"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    tr = pd.read_parquet(TRAIN)
    section("1. SHAPE / COLUMNS / LABEL DISTRIBUTION")
    print("train shape:", tr.shape)
    print("columns:", list(tr.columns))
    print("label counts:\n", tr[LABEL].value_counts(dropna=False))

    # ---- 2. Single-column determinism: does one field almost fix the label? ----
    section("2. SINGLE-COLUMN LABEL DETERMINISM (purity)")
    print("For each column: P(most common label | column value), weighted by rows.")
    print("~1.00 => that column alone almost perfectly predicts the label (leakage-like).\n")
    feature_cols = [c for c in tr.columns if c not in (LABEL, "event_id")]
    rows = []
    n = len(tr)
    for c in feature_cols:
        s = tr[c].astype("string").fillna("<NA>")
        # weighted purity = sum over groups of (max class count) / n
        grp = tr.assign(_v=s).groupby("_v")[LABEL]
        purity = grp.apply(lambda x: x.value_counts().iloc[0]).sum() / n
        nuniq = s.nunique()
        rows.append((c, purity, nuniq))
    pure = pd.DataFrame(rows, columns=["column", "weighted_purity", "n_unique"])
    pure = pure.sort_values("weighted_purity", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(pure.to_string(index=False))

    # ---- 3. message_sanitized: does the text restate the label? ----
    if "message_sanitized" in tr.columns:
        section("3. message_sanitized KEYWORD LEAKAGE")
        msg = tr["message_sanitized"].astype("string").fillna("").str.lower()
        for kw in ["malicious", "suspicious", "benign", "threat", "block",
                   "denied", "allow", "attack", "malware", "phish"]:
            hit = msg.str.contains(kw, regex=False)
            if hit.any():
                dist = tr.loc[hit, LABEL].value_counts(normalize=True).round(3).to_dict()
                print(f"  '{kw}': {hit.sum():>7} rows  label%={dist}")

    # ---- 4. Train vs Valid token / value overlap (generalization risk) ----
    section("4. TRAIN vs VALID FIELD-VALUE OVERLAP")
    print("Low overlap on a high-purity column => model memorized train-only values")
    print("=> holdout is optimistic, line score may collapse.\n")
    va = pd.read_parquet(VALID)
    print("valid shape:", va.shape)
    for c in ["src_ip", "dst_ip", "src_host", "dst_host", "username",
              "pipeline", "product_name", "vendor_name"]:
        if c in tr.columns and c in va.columns:
            tset = set(tr[c].astype("string").dropna().unique())
            vvals = va[c].astype("string").dropna()
            seen = vvals.isin(tset).mean() if len(vvals) else float("nan")
            print(f"  {c:14s} valid values seen-in-train: {seen:6.1%}  "
                  f"(train uniq={len(tset)}, valid uniq={vvals.nunique()})")

    section("VERDICT HINTS")
    top = pure.iloc[0]
    print(f"- Highest single-column purity: {top['column']} = {top['weighted_purity']:.4f}")
    if top["weighted_purity"] > 0.98:
        print("  => STRONG single-field determinism: labels are near-trivially separable.")
    else:
        print("  => No single field fully determines the label; separability is multi-field.")


if __name__ == "__main__":
    sys.exit(main())
