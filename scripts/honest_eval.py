"""Honest generalization check for the SOC baseline, via grouped CV.

The single group-split gave holdouts with tiny minority-class support (e.g. 2
malicious rows), making per-class numbers noise. This uses GroupKFold by
src_host and collects OUT-OF-FOLD predictions for every row: each row is
predicted by a model that never saw its src_host. That mirrors valid_input
(src_host overlap 0%) and yields full-support, low-variance per-class metrics.

Compares two feature sets under the SAME grouped CV:
  FULL     - build_log_documents equivalent (all identifiers)
  REDUCED  - drops raw src_ip/dst_ip/src_host/dst_host/username tokens,
             keeps /24 prefixes, port bucket, timestamp, message text,
             pipeline/product/vendor.

Also reports a random-split macro_f1 as the inflated reference.

Run: .venv/bin/python scripts/honest_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from soc_baseline.constants import LABELS
from soc_baseline.data import read_parquet_frame, stratified_sample
from soc_baseline.features import (
    _clean_token_series,
    _column_or_empty,
    _timestamp_tokens,
    ip_prefix,
    port_bucket,
)
from soc_baseline.modeling import make_model_pipeline

SAMPLE_ROWS = 200_000
RANDOM_STATE = 42
N_SPLITS = 5

FULL_CATEGORICALS = (
    "pipeline", "src_ip", "dst_ip", "src_host", "dst_host",
    "username", "product_name", "vendor_name",
)
REDUCED_CATEGORICALS = ("pipeline", "product_name", "vendor_name")


def build_documents(df: pd.DataFrame, categoricals: tuple[str, ...]) -> pd.Series:
    docs = pd.Series("", index=df.index, dtype="object")
    for column in categoricals:
        values = _clean_token_series(_column_or_empty(df, column))
        docs = docs.str.cat(column + "=" + values, sep=" ")
    src_ports = _column_or_empty(df, "src_port").map(port_bucket)
    docs = docs.str.cat("src_port_bucket=" + src_ports, sep=" ")
    src_prefixes = _column_or_empty(df, "src_ip").map(ip_prefix)
    dst_prefixes = _column_or_empty(df, "dst_ip").map(ip_prefix)
    docs = docs.str.cat("src_ip_prefix=" + src_prefixes, sep=" ")
    docs = docs.str.cat("dst_ip_prefix=" + dst_prefixes, sep=" ")
    ts = _timestamp_tokens(_column_or_empty(df, "timestamp"))
    for column in ts.columns:
        docs = docs.str.cat(column + "=" + ts[column].astype(str), sep=" ")
    messages = _column_or_empty(df, "message_sanitized").fillna("").astype(str).str.strip()
    docs = docs.str.cat(messages, sep=" ")
    return docs.str.replace(r"\s+", " ", regex=True).str.strip()


def grouped_oof(docs: pd.Series, y: pd.Series, groups: pd.Series, tag: str) -> None:
    docs = docs.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    oof = pd.Series(index=y.index, dtype="object")
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, ho) in enumerate(gkf.split(docs, y, groups), 1):
        model = make_model_pipeline(120_000, 3, RANDOM_STATE)
        model.fit(docs.iloc[tr], y.iloc[tr])
        oof.iloc[ho] = model.predict(docs.iloc[ho])
    macro = f1_score(y, oof, labels=list(LABELS), average="macro", zero_division=0)
    weighted = f1_score(y, oof, labels=list(LABELS), average="weighted", zero_division=0)
    acc = float((oof == y).mean())
    print(f"\n[{tag}]  (out-of-fold, unseen src_host)")
    print(f"  accuracy={acc:.5f}  macro_f1={macro:.5f}  weighted_f1={weighted:.5f}")
    for lab in LABELS:
        mask = y == lab
        sup = int(mask.sum())
        rec = float((oof[mask] == lab).mean()) if sup else float("nan")
        prec_mask = oof == lab
        prec = float((y[prec_mask] == lab).mean()) if prec_mask.any() else float("nan")
        print(f"    {lab:11s} support={sup:6d}  recall={rec:.4f}  precision={prec:.4f}")


def main() -> None:
    df = read_parquet_frame("data/train.parquet")
    sample = stratified_sample(df, "label_binary", SAMPLE_ROWS, RANDOM_STATE)
    y = sample["label_binary"].astype(str)
    groups = sample["src_host"].astype(str).fillna("<NA>")
    print(f"sample rows={len(sample)}  label dist={y.value_counts().to_dict()}")
    print(f"src_host unique={groups.nunique()}  -> GroupKFold({N_SPLITS})")

    docs_full = build_documents(sample, FULL_CATEGORICALS)
    docs_reduced = build_documents(sample, REDUCED_CATEGORICALS)

    # inflated reference: random split, full features
    tr_i, ho_i = train_test_split(sample.index, test_size=0.2,
                                  random_state=RANDOM_STATE, stratify=y)
    ref = make_model_pipeline(120_000, 3, RANDOM_STATE)
    ref.fit(docs_full.loc[tr_i], y.loc[tr_i])
    ref_macro = f1_score(y.loc[ho_i], ref.predict(docs_full.loc[ho_i]),
                         labels=list(LABELS), average="macro", zero_division=0)
    print(f"\n[REFERENCE random split + FULL]  macro_f1={ref_macro:.5f} (inflated)")

    grouped_oof(docs_full, y, groups, "FULL features")
    grouped_oof(docs_reduced, y, groups, "REDUCED features (identifiers dropped)")


if __name__ == "__main__":
    main()
