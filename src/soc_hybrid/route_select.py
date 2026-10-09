"""Choose the benign weight, routing triggers and fusion rule on train-internal holdouts only.

A held-out document goes to the LLM when any enabled trigger fires:

- unseen source: its source has no training rows (every LOSO document);
- novelty: its LSA similarity to the nearest training document is below s;
- new pair: the classifier's label was never seen with that source in training;
- doubt: the benign-first decision confidence is below t.

Every held-out unique document has the classifier's holdout probabilities and
its novelty; a sample also has the LLM's label (llm_holdout.py). Per holdout,
errors among routed documents are estimated from its sampled routed documents
(the LOSO/LOCO sample is uniform within a holdout, hence uniform among its
routed documents). ud5 samples target the documents a trigger can pick, so they
count as they are and other routed ud5 documents take the pooled LLM error rate.

python -m soc_hybrid.route_select --runs tfidf_word_lr_rows_cap20000,...
  → artifacts/hybrid/route_select/selection.json
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from .data import ALERTS, LABELS, OUT, events, unique_documents
from .decision import benign_first, decision_confidence
from .holdout import HOLDOUT_DIR
from .llm_holdout import RESULT as LLM_RESULT
from .metrics import BENIGN_MISS_COST
from .novelty import RESULT as NOVELTY_RESULT

SELECT_DIR = OUT / "route_select"
BENIGN_WEIGHTS = (1, 3, 10, 30, 100, 300)
DOUBT = (0.0, 0.7, 0.9, 0.99)
NOVELTY = (0.0, 0.8, 0.9, 0.95, 0.97)
FUSIONS = ("llm", "llm_gate", "agree_alert", "llm_prob", "llm_prob_gate", "llm_prob_gate_km")
LLM_WEIGHTS = (1, 3, 10, 30, 100, 300)  # benign weight applied to the LLM's own label probabilities
# Prior share of events from formats the classifier has not seen. Optimizing the
# out-of-distribution cost alone drives the benign weight to the grid's edge and
# multiplies the in-distribution cost by five, so the objective mixes the two.
PRIOR_OOD = 0.01


def fuse(rule: str, llm_label, clf_label, proba: np.ndarray, llm_proba=None, llm_weight: float = 10.0):
    """Final label of a routed document from the LLM's answer and the classifier's view.

    ``llm_prob``: the benign-first rule on the LLM's label probabilities (weight
    ``llm_weight``); ``llm_prob_gate``: the same rule decides benign or alert and the
    classifier picks the alert type.
    """

    if llm_label is None or (isinstance(llm_label, float) and np.isnan(llm_label)):
        return clf_label
    if rule == "llm_prob_gate_km" and clf_label == "malicious":
        # On train holdouts the LLM never recognizes this dataset's malicious class (0 of 391 malicious
        # documents, 109 of them called benign), so a malicious decision is not the LLM's to veto.
        return clf_label
    if rule in ("llm_prob", "llm_prob_gate", "llm_prob_gate_km"):
        if llm_proba is None:
            llm_proba = np.array([float(llm_label == label) for label in LABELS])
        decided = benign_first(np.asarray(llm_proba, dtype=float)[None, :], llm_weight)[0]
        if decided == "benign" or rule == "llm_prob":
            return decided
        return ALERTS[int(np.argmax(proba[1:]))]
    if rule == "llm":
        return llm_label
    clf_alert = ALERTS[int(np.argmax(proba[1:]))]
    if rule == "llm_gate":
        # The LLM decides benign or alert; the classifier, which knows the dataset's conventions, picks the alert type.
        return "benign" if llm_label == "benign" else clf_alert
    if rule == "agree_alert":
        return clf_label if (llm_label != "benign" and clf_label != "benign") else "benign"
    raise ValueError(rule)


def _costs(truth, pred) -> tuple[float, float]:
    truth, pred = np.asarray(truth, dtype=object), np.asarray(pred, dtype=object)
    return float((truth != pred).sum()), float(((truth == "benign") & (pred != "benign")).sum())


IN_DISTRIBUTION = ("ud5", "cl5")


def training_cells(unique: pd.DataFrame, kind: str, holdout: str, fold_of, doc_index=None) -> set:
    """(source, label) cells present in a holdout's training part. ``fold_of``: fold arrays by kind (or the ud5 array)."""

    cells = set(zip(unique["source"], unique["label"]))
    if kind == "loso":
        return {cell for cell in cells if cell[0] != holdout}
    if kind == "loco":
        source, label = holdout.rsplit("|", 1)
        return cells - {(source, label)}
    folds = fold_of[kind] if isinstance(fold_of, dict) else fold_of
    keep = folds != int(holdout.removeprefix("fold"))
    return set(zip(unique.loc[keep, "source"], unique.loc[keep, "label"]))


class Simulator:
    """Vectorized: labels as codes 0/1/2, per-holdout sums with bincount, fusion only where the LLM answered."""

    def __init__(self, run: str, variant: str = "codebook", llm_result=LLM_RESULT):
        from .splits import cluster_folds, unique_doc_folds

        unique = unique_documents(events("train"))
        probs = pd.read_parquet(HOLDOUT_DIR / run / "probs.parquet")
        novelty = pd.read_parquet(NOVELTY_RESULT)
        frame = probs.merge(novelty, on=["kind", "holdout", "doc_index"], how="left")
        llm = pd.read_parquet(llm_result).set_index(["kind", "holdout", "doc_index"])[f"llm_{variant}"]
        keys = pd.MultiIndex.from_frame(frame[["kind", "holdout", "doc_index"]])
        code = {label: i for i, label in enumerate(LABELS)}
        self.llm = np.array([code.get(label, -1) if isinstance(label, str) else -1 for label in llm.reindex(keys)])
        table = pd.read_parquet(llm_result).set_index(["kind", "holdout", "doc_index"])
        columns = [f"p_{label}_{variant}" for label in LABELS]
        if all(column in table.columns for column in columns):
            self.p_llm = table[columns].reindex(keys).to_numpy(dtype=float)
        else:  # hard labels only: a one-hot distribution
            self.p_llm = np.eye(3)[np.clip(self.llm, 0, 2)] * (self.llm >= 0)[:, None]
        self.truth = frame["label"].map(code).to_numpy()
        self.proba = frame[[f"p_{label}" for label in LABELS]].to_numpy()
        self.maxsim = frame["maxsim"].to_numpy()
        self.loso = (frame["kind"] == "loso").to_numpy()
        groups = frame["kind"] + "\0" + frame["holdout"]
        self.group, names = pd.factorize(groups)
        self.names = [name.split("\0") for name in names]
        self.n_groups = len(names)
        self.is_ud5 = np.array([kind in IN_DISTRIBUTION for kind, _ in self.names])
        self.docs = np.bincount(self.group, minlength=self.n_groups)
        # Labels each row's source was seen with in its holdout's training part.
        fold_of = {"ud5": unique_doc_folds(unique), "cl5": cluster_folds(unique)}
        allowed = np.zeros((len(frame), len(LABELS)), dtype=bool)
        for g, (kind, holdout) in enumerate(self.names):
            cells = training_cells(unique, kind, holdout, fold_of, None)
            rows = np.flatnonzero(self.group == g)
            sources = frame["source"].to_numpy()[rows]
            for i, label in enumerate(LABELS):
                allowed[rows, i] = [(src, label) in cells for src in sources]
        self.allowed = allowed
        self.alert_type = 1 + self.proba[:, 1:].argmax(axis=1)
        self._decisions = {}

    def _decision(self, w: float):
        if w not in self._decisions:
            scores = self.proba.copy()
            scores[:, 0] *= w
            self._decisions[w] = (scores.argmax(axis=1), scores.max(axis=1) / scores.sum(axis=1))
        return self._decisions[w]

    def _sum(self, mask) -> np.ndarray:
        return np.bincount(self.group, weights=mask.astype(float), minlength=self.n_groups)

    def run(self, w: float, t: float, s: float, pair: bool, unseen: bool, rule: str, wl: float = 10.0) -> pd.DataFrame:
        clf, conf = self._decision(w)
        routed = conf < t
        if s > 0:
            routed |= self.maxsim < s
        if unseen:
            routed |= self.loso
        if pair:
            routed |= ~self.allowed[np.arange(len(clf)), clf]
        answered = self.llm >= 0
        if rule == "llm":
            fused = np.where(answered, self.llm, clf)
        elif rule == "llm_gate":
            fused = np.where(answered, np.where(self.llm == 0, 0, self.alert_type), clf)
        elif rule == "agree_alert":
            fused = np.where(answered, np.where((self.llm > 0) & (clf > 0), clf, 0), clf)
        else:  # llm_prob, llm_prob_gate, llm_prob_gate_km
            scores = np.nan_to_num(self.p_llm.copy())
            scores[:, 0] *= wl
            decided = scores.argmax(axis=1)
            if rule in ("llm_prob_gate", "llm_prob_gate_km"):
                decided = np.where(decided == 0, 0, self.alert_type)
            if rule == "llm_prob_gate_km":
                decided = np.where(clf == 2, 2, decided)
            fused = np.where(answered, decided, clf)
        truth = self.truth
        known = routed & answered
        kept = ~routed
        kept_err = self._sum(kept & (truth != clf))
        kept_bm = self._sum(kept & (truth == 0) & (clf != 0))
        known_err = self._sum(known & (truth != fused))
        known_bm = self._sum(known & (truth == 0) & (fused != 0))
        n_routed = self._sum(routed)
        n_known = self._sum(known)
        pooled_err = known_err.sum() / max(n_known.sum(), 1)
        pooled_bm = known_bm.sum() / max(n_known.sum(), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(n_known > 0, n_routed / n_known, 0.0)
        ood_err = np.where(n_known > 0, known_err * scale, n_routed * pooled_err)
        ood_bm = np.where(n_known > 0, known_bm * scale, n_routed * pooled_bm)
        ud5_err = known_err + (n_routed - n_known) * pooled_err
        ud5_bm = known_bm + (n_routed - n_known) * pooled_bm
        errors = kept_err + np.where(self.is_ud5, ud5_err, ood_err)
        bm = kept_bm + np.where(self.is_ud5, ud5_bm, ood_bm)
        base_err = self._sum(truth != clf)
        base_bm = self._sum((truth == 0) & (clf != 0))
        return pd.DataFrame({
            "kind": [k for k, _ in self.names], "holdout": [h for _, h in self.names], "docs": self.docs,
            "routed": n_routed, "errors": errors, "benign_misses": bm, "cost": BENIGN_MISS_COST * bm + errors - bm,
            "clf_errors": base_err, "clf_benign_misses": base_bm, "clf_cost": BENIGN_MISS_COST * base_bm + base_err - base_bm,
        })


def aggregate(per_holdout: pd.DataFrame) -> dict:
    """ud5 pooled over documents. loso and loco two ways: every holdout weighs the same (macro), or every
    held-out document does (pooled). Holdouts range from 1 to 136,826 documents, so the macro view can
    turn on a 5-document cell and the pooled view on Active Directory alone; selection uses both."""

    out = {}
    for kind, group in per_holdout.groupby("kind"):
        pooled = group["cost"].sum() / group["docs"].sum()
        if kind in IN_DISTRIBUTION:
            out[kind] = {"cost_per_doc": pooled, "clf_cost_per_doc": group["clf_cost"].sum() / group["docs"].sum(),
                         "routed_share": group["routed"].sum() / group["docs"].sum(), "routed": int(group["routed"].sum())}
        else:
            out[kind] = {"cost_per_doc": float((group["cost"] / group["docs"]).mean()), "pooled_cost_per_doc": float(pooled),
                         "clf_cost_per_doc": float((group["clf_cost"] / group["docs"]).mean()),
                         "clf_pooled_cost_per_doc": float(group["clf_cost"].sum() / group["docs"].sum()),
                         "routed_share": float(group["routed"].sum() / group["docs"].sum()), "routed": int(group["routed"].sum())}
    out["ood_cost_per_doc"] = float(np.mean([out[k]["cost_per_doc"] for k in ("loso", "loco")]))
    out["ood_pooled_cost_per_doc"] = float(np.mean([out[k]["pooled_cost_per_doc"] for k in ("loso", "loco")]))
    return out


def objective(row, prior_ood: float = PRIOR_OOD, pooled: bool = False, indist: str = "ud5") -> float:
    ood = row["ood_pooled_cost_per_doc"] if pooled else row["ood_cost_per_doc"]
    return (1 - prior_ood) * row[f"{indist}_cost_per_doc"] + prior_ood * ood


def flat(result: dict) -> dict:
    row = {"ood_cost_per_doc": result["ood_cost_per_doc"], "ood_pooled_cost_per_doc": result["ood_pooled_cost_per_doc"]}
    for kind in ("ud5", "cl5", "loso", "loco"):
        for metric, value in result.get(kind, {}).items():
            row[f"{kind}_{metric}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--variant", default="codebook")
    parser.add_argument("--max-ud5-routed", type=float, default=0.01)
    parser.add_argument("--prior-ood", type=float, default=PRIOR_OOD)
    parser.add_argument("--indist", default="ud5", choices=IN_DISTRIBUTION,
                        help="in-distribution proxy: ud5 (iteration 1) or cl5 (iteration 2)")
    parser.add_argument("--fusions", default=",".join(FUSIONS))
    parser.add_argument("--llm-result", default=str(LLM_RESULT))
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--grid-only", action="store_true", help="only write this run's grid (for parallel runs)")
    parser.add_argument("--from-grids", default=None, help="comma-separated grid parquet files to select from")
    args = parser.parse_args()
    grid = []
    runs = [] if args.from_grids else args.runs.split(",")
    for run in runs:
        simulator = Simulator(run, args.variant, args.llm_result)
        fusions = tuple(args.fusions.split(","))
        for w, t, s, pair, unseen, rule in itertools.product(BENIGN_WEIGHTS, DOUBT, NOVELTY, (False, True), (False, True), fusions):
            no_trigger = t == 0 and s == 0 and not pair and not unseen
            if no_trigger and rule != "llm":
                continue
            for wl in (LLM_WEIGHTS if rule.startswith("llm_prob") else (None,)):
                result = aggregate(simulator.run(w, t, s, pair, unseen, rule, wl or 10.0))
                grid.append({"run": run, "w": w, "t": t, "s": s, "pair": pair, "unseen": unseen,
                             "fusion": "none" if no_trigger else rule, "wl": wl, **flat(result)})
    table = pd.DataFrame(grid) if not args.from_grids else pd.concat(
        [pd.read_parquet(path) for path in args.from_grids.split(",")], ignore_index=True)
    if args.grid_only:
        SELECT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_parquet(SELECT_DIR / f"grid_{args.variant}{args.tag}.parquet", index=False)
        return
    table = table.drop(columns=[c for c in ("objective", "objective_pooled", "regret") if c in table.columns])
    table["objective"] = objective(table, args.prior_ood, indist=args.indist)
    table["objective_pooled"] = objective(table, args.prior_ood, pooled=True, indist=args.indist)
    feasible = table[table[f"{args.indist}_routed_share"] <= args.max_ud5_routed].copy()
    # Minimax regret over the two aggregations: within the smallest factor of the best on both.
    feasible["regret"] = np.maximum(feasible["objective"] / feasible["objective"].min(),
                                    feasible["objective_pooled"] / feasible["objective_pooled"].min())
    table = table.join(feasible["regret"])
    order = ["regret", "objective_pooled"]
    best = feasible.sort_values(order).iloc[0].to_dict()
    without_llm = feasible[feasible["fusion"] == "none"].sort_values(order).iloc[0].to_dict()
    sensitivity = {"macro_only": feasible.sort_values(["objective", "objective_pooled"]).iloc[0].to_dict(),
                   "pooled_only": feasible.sort_values(["objective_pooled", "objective"]).iloc[0].to_dict()}
    for prior in (0.001, 0.05):
        macro = objective(feasible, prior, indist=args.indist)
        pooled = objective(feasible, prior, True, indist=args.indist)
        ranked = feasible.assign(o=np.maximum(macro / macro.min(), pooled / pooled.min()))
        sensitivity[f"prior_{prior}"] = ranked.sort_values(["o", "objective_pooled"]).iloc[0].drop("o").to_dict()
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(SELECT_DIR / f"grid_{args.variant}{args.tag}.parquet", index=False)
    report = {"variant": args.variant, "indist": args.indist, "max_routed": args.max_ud5_routed, "prior_ood": args.prior_ood,
              "best": best, "best_without_llm": without_llm, "sensitivity": sensitivity}
    (SELECT_DIR / f"selection_{args.variant}{args.tag}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    cols = ["run", "w", "t", "s", "pair", "unseen", "fusion", "wl", "regret", "objective", "objective_pooled", "ood_cost_per_doc", "loso_cost_per_doc", "loco_cost_per_doc",
            f"{args.indist}_cost_per_doc", f"{args.indist}_routed_share", "loso_routed_share", "loco_routed_share"]
    pd.set_option("display.width", 250)
    print(feasible.sort_values(order)[cols].head(20).round(5).to_string())
    print("\nbest:", {k: best[k] for k in cols})
    print("best without LLM:", {k: without_llm[k] for k in cols})
    for prior, row in sensitivity.items():
        print(f"prior {prior}:", {k: row[k] for k in cols})


if __name__ == "__main__":
    main()
