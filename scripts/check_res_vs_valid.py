"""Does the deployed baseline (res.csv) actually predict suspicious/malicious on
valid_input via the generalizable product_name/vendor signal? This is the real
train->valid behavior, not a within-train CV proxy.
"""
from __future__ import annotations
import pandas as pd
from soc_baseline.data import read_parquet_frame

res = pd.read_csv("res.csv", dtype={"event_id": str})
va = read_parquet_frame("data/valid_input.parquet",
                        columns=["event_id", "product_name", "vendor_name", "pipeline"])
va["event_id"] = va["event_id"].astype(str)
m = va.merge(res, on="event_id", how="left")
print("res.csv pred_label distribution:")
print(m["pred_label"].value_counts(dropna=False), "\n")

for field in ["product_name", "vendor_name"]:
    print("=" * 60)
    print(f"pred_label by {field} (rows where model predicts non-benign or top products)")
    print("=" * 60)
    ct = pd.crosstab(m[field].fillna("<NA>"), m["pred_label"])
    ct["total"] = ct.sum(axis=1)
    # show products that receive any suspicious/malicious prediction
    keep = ct.index[(ct.get("suspicious", 0) > 0) | (ct.get("malicious", 0) > 0)]
    show = ct.loc[keep].sort_values("total", ascending=False)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(show, "\n")
