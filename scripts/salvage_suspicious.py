"""Is `suspicious` (and `malicious`) recoverable from GENERALIZABLE signal, or
is it keyed to entity identity (and thus hopeless for valid_input's new hosts)?

Three probes on a stratified sample of train.parquet:

  1. Mutual information (normalized) between each field and each binary target
     (is_suspicious / is_malicious). ~0 for every generalizable field => no
     content signal.
  2. Message-only binary classifier under GroupKFold(src_host): the recall the
     model text alone can achieve for unseen hosts.
  3. For the strongest suspicious-flagging values of generalizable fields:
     host-spread (does the signal span many hosts = real, or 1 host = proxy?)
     and whether the value appears in valid_input at all.

Run: .venv/bin/python scripts/salvage_suspicious.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GroupKFold

from soc_baseline.data import read_parquet_frame, stratified_sample
from soc_baseline.features import _column_or_empty, ip_prefix, port_bucket

SAMPLE_ROWS = 200_000
RANDOM_STATE = 42

# fields grouped by their train<->valid overlap (from check_leakage.py)
GENERALIZABLE = ["pipeline", "product_name", "vendor_name", "dst_ip",
                 "dst_ip_prefix", "src_ip_prefix", "src_port_bucket", "dst_host"]
LOW_OVERLAP = ["src_host", "src_ip", "username"]  # for contrast


def derived(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["dst_ip_prefix"] = _column_or_empty(sample, "dst_ip").map(ip_prefix)
    out["src_ip_prefix"] = _column_or_empty(sample, "src_ip").map(ip_prefix)
    out["src_port_bucket"] = _column_or_empty(sample, "src_port").map(port_bucket)
    return out


def norm_mi(feature: pd.Series, target: np.ndarray) -> float:
    """MI(feature; target) normalized by target entropy -> [0,1]."""
    f = feature.astype("string").fillna("<NA>")
    mi = mutual_info_score(target, f)
    h = entropy(np.bincount(target) / len(target))
    return float(mi / h) if h > 0 else 0.0


def message_only_recall(sample: pd.DataFrame, target: np.ndarray, groups: pd.Series,
                        name: str) -> None:
    msg = _column_or_empty(sample, "message_sanitized").fillna("").astype(str)
    msg = msg.reset_index(drop=True)
    y = pd.Series(target).reset_index(drop=True)
    grp = groups.reset_index(drop=True)
    oof = np.zeros(len(y), dtype=int)
    gkf = GroupKFold(n_splits=5)
    for tr, ho in gkf.split(msg, y, grp):
        vec = TfidfVectorizer(lowercase=True, max_features=100_000, min_df=3,
                              ngram_range=(1, 2), sublinear_tf=True)
        xt = vec.fit_transform(msg.iloc[tr])
        clf = SGDClassifier(loss="log_loss", class_weight="balanced",
                            max_iter=40, tol=1e-3, random_state=RANDOM_STATE)
        clf.fit(xt, y.iloc[tr])
        oof[ho] = clf.predict(vec.transform(msg.iloc[ho]))
    tp = int(((oof == 1) & (y == 1)).sum())
    pos = int((y == 1).sum())
    pred_pos = int((oof == 1).sum())
    recall = tp / pos if pos else float("nan")
    prec = tp / pred_pos if pred_pos else float("nan")
    print(f"  message-only, unseen host -> {name}: recall={recall:.3f} "
          f"precision={prec:.3f} (support={pos}, predicted_pos={pred_pos})")


def top_values(sample: pd.DataFrame, field: str, target: np.ndarray,
               valid_vals: set, min_count: int = 30) -> None:
    f = sample[field].astype("string").fillna("<NA>")
    dfp = pd.DataFrame({"v": f, "t": target, "host": sample["src_host"].astype("string")})
    g = dfp.groupby("v")
    agg = g.agg(rate=("t", "mean"), count=("t", "size"), hosts=("host", "nunique"))
    agg = agg[agg["count"] >= min_count].sort_values("rate", ascending=False).head(6)
    print(f"  [{field}] top values by positive-rate (min_count={min_count}):")
    for v, row in agg.iterrows():
        in_valid = "yes" if str(v) in valid_vals else "NO"
        print(f"      rate={row['rate']:.3f} count={int(row['count']):6d} "
              f"hosts={int(row['hosts']):3d} in_valid={in_valid}  value={str(v)[:40]}")


def main() -> None:
    df = read_parquet_frame("data/train.parquet")
    sample = derived(stratified_sample(df, "label_binary", SAMPLE_ROWS, RANDOM_STATE))
    y = sample["label_binary"].astype(str).to_numpy()
    is_susp = (y == "suspicious").astype(int)
    is_mal = (y == "malicious").astype(int)
    groups = sample["src_host"].astype(str).fillna("<NA>")

    print("=" * 70)
    print("1. NORMALIZED MUTUAL INFORMATION  (fraction of target entropy explained)")
    print("=" * 70)
    print(f"{'field':16s} {'MI->suspicious':>16s} {'MI->malicious':>16s}  overlap")
    for field in GENERALIZABLE + LOW_OVERLAP:
        tag = "GEN" if field in GENERALIZABLE else "low-overlap"
        print(f"{field:16s} {norm_mi(sample[field], is_susp):16.4f} "
              f"{norm_mi(sample[field], is_mal):16.4f}  {tag}")

    print("\n" + "=" * 70)
    print("2. MESSAGE-TEXT-ONLY BINARY CLASSIFIER, unseen host (GroupKFold)")
    print("=" * 70)
    message_only_recall(sample, is_susp, groups, "suspicious")
    message_only_recall(sample, is_mal, groups, "malicious")

    print("\n" + "=" * 70)
    print("3. STRONGEST GENERALIZABLE suspicious-flagging VALUES (spread + in valid?)")
    print("=" * 70)
    va = read_parquet_frame("data/valid_input.parquet")
    va_dvals = {c: set(va[c].astype("string").dropna().unique()) if c in va.columns else set()
                for c in ["pipeline", "product_name", "vendor_name", "dst_ip", "dst_host"]}
    for field in ["pipeline", "product_name", "vendor_name", "dst_ip", "dst_host"]:
        top_values(sample, field, is_susp, va_dvals.get(field, set()))


if __name__ == "__main__":
    main()
