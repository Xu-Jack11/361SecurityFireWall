"""Ablations on valid that reuse one run's components (no new LLM calls), plus a table across runs.

python -m soc_hybrid.ablation --base final --runs abl_w1,abl_random,...  → artifacts/hybrid/ablation.json
(ablation_cost.json when the base run uses the corrected rule, --miss-weight)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, OUT
from .decision import benign_first
from .evaluate import row_frame
from .metrics import confusion_frame, evaluate, summary_row
from .pipeline import RUNS
from .route_select import fuse


def derived_variants(run: str) -> list[tuple[str, dict]]:
    """Variants computed from one run's saved components."""

    config = json.loads((RUNS / run / "config.json").read_text())
    rows = row_frame(run)
    proba = rows[[f"p_{label}" for label in LABELS]].to_numpy()
    llm_proba = rows[[f"llm_p_{label}" for label in LABELS]].to_numpy(dtype=float)
    routed = rows["routed"].to_numpy(dtype=bool)
    out = [(f"{run} (system)", evaluate(rows["truth"], rows["final"])),
           ("classifier only (same w)", evaluate(rows["truth"], rows["clf"])),
           ("classifier only, w = 1", evaluate(rows["truth"], benign_first(proba, 1.0)))]

    def fused(rule, wl=None):
        pred = rows["clf"].to_numpy(dtype=object).copy()
        for i in np.flatnonzero(routed):
            q = llm_proba[i]
            pred[i] = fuse(rule, rows["llm"].iat[i], pred[i], proba[i], None if np.isnan(q).any() else q,
                           wl if wl is not None else config["llm_weight"])
        return pred

    # No benign priority anywhere: plain argmax for the classifier (w = 1) and the LLM (wl = 1).
    clf_w1 = benign_first(proba, 1.0)
    pred = clf_w1.copy()
    for i in np.flatnonzero(routed):
        q = llm_proba[i]
        pred[i] = fuse(config["fusion"], rows["llm"].iat[i], clf_w1[i], proba[i], None if np.isnan(q).any() else q, 1.0)
    out.append(("no benign priority (w = 1, wl = 1)", evaluate(rows["truth"], pred)))
    # Post hoc: plain argmax for the classifier, benign-first LLM (wl as configured).
    pred = clf_w1.copy()
    for i in np.flatnonzero(routed):
        q = llm_proba[i]
        pred[i] = fuse(config["fusion"], rows["llm"].iat[i], clf_w1[i], proba[i], None if np.isnan(q).any() else q,
                       config["llm_weight"])
    out.append(("post hoc: classifier w = 1, LLM wl as configured", evaluate(rows["truth"], pred)))
    for rule in ("llm", "llm_gate", "agree_alert"):
        out.append((f"fusion {rule}", evaluate(rows["truth"], fused(rule))))
    for wl in (1, 3, 10, 30, 100, 300):
        out.append((f"fusion {config['fusion']}, wl = {wl}", evaluate(rows["truth"], fused(config["fusion"], wl))))
    out.append(("fusion llm_prob (LLM picks the alert type), wl as configured", evaluate(rows["truth"], fused("llm_prob"))))

    # Smaller LLM budgets: the routed documents are a priority-ordered prefix (unseen source, then least
    # similar, then least sure), so a budget of B keeps the first B of them and the rest fall back to the classifier.
    docs = pd.read_parquet(RUNS / run / "docs.parquet")
    routed_docs = docs[docs["routed"]]
    order = np.lexsort((routed_docs["conf"].to_numpy(), routed_docs["maxsim"].to_numpy(),
                        ~routed_docs["trigger_unseen_source"].to_numpy()))
    ranked = routed_docs.index.to_numpy()[order]
    position = pd.Series(np.arange(len(ranked)), index=ranked)
    doc_position = position.reindex(np.arange(len(docs))).to_numpy()
    # row_frame keeps valid_documents() order, so map each event to its document's position via the doc index.
    from .evaluate import KEYS
    from .pipeline import valid_documents
    from .data import events

    keys = valid_documents()[KEYS].assign(doc_index=np.arange(len(docs)))
    event_doc = events("valid")[["event_id"] + KEYS].merge(keys, on=KEYS, how="left")["doc_index"].to_numpy()
    for budget in (250, 500, 1000):
        keep = doc_position[event_doc] < budget
        pred = np.where(keep, rows["final"].to_numpy(dtype=object), rows["clf"].to_numpy(dtype=object))
        out.append((f"LLM budget {budget} documents", evaluate(rows["truth"], pred)))
    return out


def budget_positions(run: str, config: dict) -> np.ndarray:
    """Each valid event's document position in the routing order (NaN when not routed), so that a budget of B
    documents keeps the first B routed documents and the rest fall back to the classifier."""

    from .data import events
    from .evaluate import KEYS
    from .pipeline import valid_documents

    docs = pd.read_parquet(RUNS / run / "docs.parquet")
    routed_docs = docs[docs["routed"]]
    first, second = ((routed_docs["maxsim"], routed_docs["conf"]) if config.get("priority", "novelty") == "novelty"
                     else (routed_docs["conf"], routed_docs["maxsim"]))
    order = np.lexsort((second.to_numpy(), first.to_numpy(), ~routed_docs["trigger_unseen_source"].to_numpy()))
    position = pd.Series(np.arange(len(order)), index=routed_docs.index.to_numpy()[order])
    doc_position = position.reindex(np.arange(len(docs))).to_numpy()
    keys = valid_documents()[KEYS].assign(doc_index=np.arange(len(docs)))
    event_doc = events("valid")[["event_id"] + KEYS].merge(keys, on=KEYS, how="left")["doc_index"].to_numpy()
    return doc_position[event_doc]


def derived_cost_variants(run: str) -> list[tuple[str, dict]]:
    """Variants of a run made under the corrected rule (--miss-weight), from its saved components."""

    from .decision import cost_matrix, fuse_codes, min_cost

    config = json.loads((RUNS / run / "config.json").read_text())
    rows = row_frame(run)
    labels = np.asarray(LABELS, dtype=object)
    code = {label: i for i, label in enumerate(LABELS)}
    proba = rows[[f"p_{label}" for label in LABELS]].to_numpy()
    llm_proba = rows[[f"llm_p_{label}" for label in LABELS]].to_numpy(dtype=float)
    answered = rows["routed"].to_numpy(dtype=bool) & rows["llm"].isin(LABELS).to_numpy()
    onehot = np.eye(3)[rows.loc[answered, "llm"].map(code).to_numpy()]
    q = np.where(np.isnan(llm_proba[answered]).any(axis=1, keepdims=True), onehot, llm_proba[answered])
    m_dec, truth = config["miss_weight"], rows["truth"]
    clf = rows["clf"].map(code).to_numpy()

    def fused(rule, param, base=clf, m=m_dec):
        pred = base.copy()
        pred[answered] = fuse_codes(rule, param, base[answered], proba[answered], q, m)
        return labels[pred]

    out = [(f"{run} (system)", evaluate(truth, rows["final"])),
           ("classifier only (same miss weight)", evaluate(truth, rows["clf"]))]
    for m in (1, 2, 3, 5, 10, 20):
        out.append((f"classifier only, miss weight {m}", evaluate(truth, min_cost(proba, cost_matrix(m)))))
    out.append(("classifier only, benign-first w = 10 (original rule)", evaluate(truth, benign_first(proba, 10.0))))
    for m in (1, 2, 3, 5, 10, 20):
        base = (proba @ cost_matrix(m)).argmin(axis=1)
        out.append((f"system at classifier miss weight {m}", evaluate(truth, fused(config["fusion"], config["fusion_param"], base, m))))
    for rule, values in (("gate_km", (1, 2, 5, 10, 20)), ("raise", (1, 2, 5, 10, 20)), ("mix_km", (0.25, 0.5, 0.75, 1.0))):
        for value in values:
            out.append((f"fusion {rule}({value:g})", evaluate(truth, fused(rule, value))))
    # The LLM decides benign or alert even on the classifier's malicious documents (no keep-malicious).
    pred = clf.copy()
    llm_alert = (q @ cost_matrix(config["fusion_param"])).argmin(axis=1) != 0
    pred[answered] = np.where(llm_alert, 1 + proba[answered, 1:].argmax(axis=1), 0)
    out.append(("fusion gate without keep-malicious", evaluate(truth, labels[pred])))
    pred = rows["clf"].to_numpy(dtype=object).copy()
    pred[answered] = rows.loc[answered, "llm"].to_numpy(dtype=object)
    out.append(("fusion: LLM label as is", evaluate(truth, pred)))
    position = budget_positions(run, config)
    for budget in (250, 500, 1000):
        keep = position < budget
        out.append((f"LLM budget {budget} documents",
                    evaluate(truth, np.where(keep, rows["final"].to_numpy(dtype=object), rows["clf"].to_numpy(dtype=object)))))
    return out


def llm_only(base: str) -> list[tuple[str, dict]]:
    """The LLM alone on a uniform sample of distinct valid documents, against the base run on the same events."""

    path = RUNS / "llm_only_sample" / "docs.parquet"
    if not path.exists():
        return []
    sample = pd.read_parquet(path)
    keys = ["doc", "pipeline", "vendor_name", "product_name"]
    from .data import events

    rows = row_frame(base)
    va = events("valid")[["event_id"] + keys + ["llm_message"]]
    picked = va.merge(sample[keys + ["llm_message", "llm"]], on=keys + ["llm_message"], how="inner")
    rows = rows.set_index("event_id").loc[picked["event_id"]]
    return [(f"LLM alone on {len(sample)} sampled documents", evaluate(rows["truth"], picked["llm"].fillna("benign").to_numpy())),
            (f"{base} on the same documents", evaluate(rows["truth"], rows["final"])),
            (f"classifier on the same documents", evaluate(rows["truth"], rows["clf"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="final")
    parser.add_argument("--runs", default="")
    parser.add_argument("--output", default=None,
                        help="result JSON (default: ablation.json, or ablation_cost.json for a corrected-rule base)")
    args = parser.parse_args()
    table, details = [], {}
    cost_rule = json.loads((RUNS / args.base / "config.json").read_text()).get("miss_weight") is not None
    for name, result in (derived_cost_variants if cost_rule else derived_variants)(args.base):
        table.append(summary_row(name, result))
        details[name] = result
    for name, result in llm_only(args.base):
        table.append(summary_row(name, result))
        details[name] = result
    for run in [r for r in args.runs.split(",") if r]:
        rows = row_frame(run)
        result = evaluate(rows["truth"], rows["final"])
        config = json.loads((RUNS / run / "config.json").read_text())
        table.append({**summary_row(run, result), "routed_docs": config["routed_docs"]})
        details[run] = result
    frame = pd.DataFrame(table)
    output = Path(args.output) if args.output else OUT / ("ablation_cost.json" if cost_rule else "ablation.json")
    output.write_text(json.dumps({"table": table, "details": details}, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))
    print(confusion_frame(details[f"{args.base} (system)"]).to_string())


if __name__ == "__main__":
    main()
