"""Is res.csv's 44% malicious on valid defensible, or a product-mix shift artifact?
Compare train per-product label rates + volume share vs valid volume share.
"""
from __future__ import annotations
import pandas as pd
from soc_baseline.data import read_parquet_frame

tr = read_parquet_frame("data/train.parquet",
                        columns=["product_name", "vendor_name", "label_binary"])
va = read_parquet_frame("data/valid_input.parquet", columns=["product_name", "vendor_name"])
res = pd.read_csv("res.csv", dtype={"event_id": str})

def report(field: str) -> None:
    print("=" * 78)
    print(f"{field}: TRAIN label rates & share   vs   VALID share   vs   res.csv pred share")
    print("=" * 78)
    t = tr.copy(); t[field] = t[field].fillna("<NA>")
    lab = pd.crosstab(t[field], t["label_binary"], normalize="index").round(3)
    cnt = t[field].value_counts()
    tr_share = (cnt / len(t)).round(3)
    va_share = (va[field].fillna("<NA>").value_counts() / len(va)).round(3)
    out = lab.copy()
    out["train_share"] = tr_share
    out["valid_share"] = va_share.reindex(out.index).fillna(0.0)
    out = out.sort_values("valid_share", ascending=False).head(12)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(out.fillna(0.0), "\n")

report("product_name")
report("vendor_name")

print("train label prior:", tr["label_binary"].value_counts(normalize=True).round(4).to_dict())
print("res.csv pred prior:", res["pred_label"].value_counts(normalize=True).round(4).to_dict())
