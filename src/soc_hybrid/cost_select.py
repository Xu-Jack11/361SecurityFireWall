"""Selection under the corrected scoring rule, on train-internal holdouts only.

The rule: a threat (suspicious or malicious) called benign is the costliest error, suspicious and malicious
called each other earn partial credit. As a cost matrix (rows true, columns predicted; benign, suspicious,
malicious) that is [[0, 1, 1], [m, 0, 0.5], [m, 0.5, 0]]. The miss cost m is not known, so every candidate is
costed at m = 2, 5 and 10 and the choice minimizes the worst regret over the three (and over the two ways of
aggregating the out-of-distribution holdouts, as in route_select.py).

The classifier decides by minimum expected cost under its own decision matrix (miss weight m_dec, chosen here),
which equals the true m only for calibrated probabilities.

python -m soc_hybrid.cost_select classifiers  → artifacts/hybrid/cost_select/classifiers.csv
python -m soc_hybrid.cost_select picks        → cl5_extra.parquet (cl5 documents the new triggers pick, for llm_holdout)
python -m soc_hybrid.cost_select routing      → routing_grid.parquet, routing.json
python -m soc_hybrid.cost_select robust       → robust.csv, robust.json (the choice under fold aggregations that
                                                 keep tiny folds from deciding it)
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from .classifier_select import CNN, TFIDF, _load
from .data import LABELS, OUT, events, unique_documents
from .decision import cost_confidence, cost_matrix, fuse_codes
from .llm_holdout import RESULT as LLM_RESULT
from .metrics import MISS_COSTS
from .novelty import RESULT as NOVELTY_RESULT
from .route_select import PRIOR_OOD, training_cells

SELECT_DIR = OUT / "cost_select"
DECISION_MISS = (1, 2, 3, 5, 10, 20, 50, 100)
KINDS = ("cl5", "loso", "loco")
# Stage 2 candidates: the best three classifiers of stage 1 (by worst regret, one row per classifier).
STAGE2 = ("tfidf_word_lr_rows_cap20000", f"mean({CNN}, tfidf_word_lr_rows_cap20000)",
          f"mean({CNN}, tfidf_word_lr_doc_cap20000)")
ROUTE_MISS = (1, 2, 3, 5, 10)
DOUBT = (0.0, 0.7, 0.9, 0.95, 0.99)
NOVELTY = (0.0, 0.8, 0.9, 0.95)
LLM_MISS = (1, 2, 5, 10, 20)  # miss weight of the LLM's own benign-or-alert decision
MIX = (0.25, 0.5, 0.75)
# Share of cl5 documents the LLM may see. It was 0.02 (2,000 of 93,755 valid documents, the T4's time budget)
# until the time limit was lifted; the 0.02 selection is reproduced with --max-routed 0.02.
MAX_ROUTED = 1.0
# Uniform LLM samples inside the bands the wider triggers add, so routed documents there are not costed
# from the least-sure picks alone: (classifier decision miss weights, confidence bands, novelty bands).
BAND_MISS = (1, 2, 3)
CONFIDENCE_BANDS = ((0.9, 0.95), (0.95, 0.99))
NOVELTY_BANDS = ((0.8, 0.9), (0.9, 0.95))
BAND_DOCS = 50
CONFIDENCE_EDGES = (0.7, 0.9, 0.95, 0.99)  # strata for costing routed cl5 documents without an LLM answer


class HoldoutCosts:
    """Per-holdout confusion matrices of label codes, costed under any matrix."""

    def __init__(self, kinds: np.ndarray, holdouts: np.ndarray, truth: np.ndarray):
        self.group, names = pd.factorize(pd.Series(kinds) + "\0" + pd.Series(holdouts))
        self.names = [name.split("\0") for name in names]
        self.kind = np.array([kind for kind, _ in self.names])
        self.truth = truth
        self.docs = np.bincount(self.group, minlength=len(names)).astype(float)

    def confusion(self, pred: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """(groups, 3, 3) counts; ``weight`` lets estimated documents count fractionally."""

        cells = self.group * 9 + self.truth * 3 + pred
        counts = np.bincount(cells, weights=weight, minlength=len(self.names) * 9)
        return counts.reshape(len(self.names), 3, 3)

    def aggregate(self, confusion: np.ndarray, miss: float, prior_ood: float = PRIOR_OOD) -> dict:
        cost = (confusion * cost_matrix(miss)).sum(axis=(1, 2))
        out = {}
        for kind in KINDS:
            mask = self.kind == kind
            out[f"{kind}_pooled"] = cost[mask].sum() / self.docs[mask].sum()
            out[f"{kind}_macro"] = float((cost[mask] / self.docs[mask]).mean())
        for agg in ("macro", "pooled"):
            ood = (out[f"loso_{agg}"] + out[f"loco_{agg}"]) / 2
            out[f"objective_{agg}"] = (1 - prior_ood) * out["cl5_pooled"] + prior_ood * ood
        return out


def error_counts(holdout: HoldoutCosts, confusion: np.ndarray, kind: str) -> dict:
    total = confusion[holdout.kind == kind].sum(axis=0)
    docs = total.sum()
    return {f"{kind}_threat_miss_rate": total[1:, 0].sum() / docs, f"{kind}_false_alarm_rate": total[0, 1:].sum() / docs,
            f"{kind}_swap_rate": (total[1, 2] + total[2, 1]) / docs}


def add_regret(table: pd.DataFrame) -> pd.DataFrame:
    """Regret = objective / best objective, per miss cost and aggregation; ``regret`` = the worst of all."""

    columns = []
    for miss in MISS_COSTS:
        per_miss = []
        for agg in ("macro", "pooled"):
            column = f"m{miss:g}_objective_{agg}"
            ratio = table[column] / table[column].min()
            per_miss.append(ratio)
            columns.append(ratio)
        table[f"regret_m{miss:g}"] = np.maximum(*per_miss)
    table["regret"] = np.max(np.vstack(columns), axis=0)
    return table


def classifiers() -> pd.DataFrame:
    """Each candidate classifier at each decision miss weight, without the LLM."""

    base = _load(CNN)
    truth = base["label"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    keep = base["kind"].isin(KINDS).to_numpy()
    holdout = HoldoutCosts(base["kind"].to_numpy()[keep], base["holdout"].to_numpy()[keep], truth[keep])
    probs = {}
    for run in (CNN, *TFIDF):
        frame = _load(run)
        assert (frame["doc_index"].to_numpy() == base["doc_index"].to_numpy()).all()
        probs[run] = frame[[f"p_{label}" for label in LABELS]].to_numpy()[keep]
    candidates = dict(probs)
    for run in TFIDF:
        candidates[f"mean({CNN}, {run})"] = (probs[CNN] + probs[run]) / 2
    rows = []
    for name, proba in candidates.items():
        for m_dec in DECISION_MISS:
            pred = (proba @ cost_matrix(m_dec)).argmin(axis=1)
            confusion = holdout.confusion(pred)
            row = {"classifier": name, "m_dec": m_dec}
            for miss in MISS_COSTS:
                for key, value in holdout.aggregate(confusion, miss).items():
                    row[f"m{miss:g}_{key}"] = value
            for kind in KINDS:
                row.update(error_counts(holdout, confusion, kind))
            rows.append(row)
    table = add_regret(pd.DataFrame(rows)).sort_values("regret")
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SELECT_DIR / "classifiers.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["classifier", "m_dec", "regret", *[f"regret_m{m:g}" for m in MISS_COSTS],
            "cl5_threat_miss_rate", "cl5_false_alarm_rate", "cl5_swap_rate", "loso_threat_miss_rate", "loco_threat_miss_rate"]
    print(table[show].head(15).round(5).to_string(index=False))
    for miss in MISS_COSTS:
        best = table.sort_values([f"regret_m{miss:g}", "regret"]).iloc[0]
        print(f"best if m = {miss:g}: {best['classifier']} at m_dec = {best['m_dec']} (regret {best[f'regret_m{miss:g}']:.3f})")
    return table


def holdout_frame(spec: str) -> pd.DataFrame:
    """cl5 / LOSO / LOCO holdout probabilities of a run, or of ``mean(a, b)``, the average of two runs."""

    runs = spec[5:-1].split(", ") if spec.startswith("mean(") else [spec]
    frames = [_load(run) for run in runs]
    frame = frames[0].copy()
    columns = [f"p_{label}" for label in LABELS]
    for other in frames[1:]:
        assert (other["doc_index"].to_numpy() == frame["doc_index"].to_numpy()).all()
    frame[columns] = np.mean([f[columns].to_numpy() for f in frames], axis=0)
    return frame[frame["kind"].isin(KINDS)].reset_index(drop=True)


def picks(docs: int = 100) -> pd.DataFrame:
    """cl5 documents the cost-based triggers can pick, per stage-2 classifier and decision miss weight: the least
    sure, the first new (source, label) pairs, and uniform samples inside the confidence and novelty bands that the
    wider triggers add. The LLM answers them in llm_holdout.py (--cl5-extra)."""

    from .splits import cluster_folds, unique_doc_folds

    unique = unique_documents(events("train"))
    fold_of = {"ud5": unique_doc_folds(unique), "cl5": cluster_folds(unique)}
    picked = []
    for spec in STAGE2:
        part = holdout_frame(spec)
        part = part[part["kind"] == "cl5"].copy()
        proba = part[[f"p_{label}" for label in LABELS]].to_numpy()
        cells = {h: training_cells(unique, "cl5", h, fold_of) for h in part["holdout"].unique()}
        for m_dec in ROUTE_MISS:
            costs = cost_matrix(m_dec)
            part["conf"] = cost_confidence(proba, costs)
            picked.append(part.nsmallest(docs, "conf")[["holdout", "doc_index"]])
            part["clf"] = np.asarray(LABELS, dtype=object)[(proba @ costs).argmin(axis=1)]
            new = [(src, lab) not in cells[h] for h, src, lab in zip(part["holdout"], part["source"], part["clf"])]
            picked.append(part.loc[new, ["holdout", "doc_index"]].head(docs))
            if m_dec in BAND_MISS:
                for lo, hi in CONFIDENCE_BANDS:
                    band = part[(part["conf"] >= lo) & (part["conf"] < hi)]
                    picked.append(band.sample(min(BAND_DOCS, len(band)), random_state=0)[["holdout", "doc_index"]])
    novelty = pd.read_parquet(NOVELTY_RESULT)
    novelty = novelty[novelty["kind"] == "cl5"]
    for lo, hi in NOVELTY_BANDS:
        band = novelty[(novelty["maxsim"] >= lo) & (novelty["maxsim"] < hi)]
        picked.append(band.sample(min(BAND_DOCS, len(band)), random_state=0)[["holdout", "doc_index"]])
    plan = pd.concat(picked).drop_duplicates().reset_index(drop=True)
    if LLM_RESULT.exists():
        done = pd.read_parquet(LLM_RESULT, columns=["kind", "holdout", "doc_index"])
        done = set(zip(done.loc[done["kind"] == "cl5", "holdout"], done.loc[done["kind"] == "cl5", "doc_index"]))
        print(f"{len(plan)} picked cl5 documents, {sum((h, d) in done for h, d in zip(plan['holdout'], plan['doc_index']))} already answered")
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(SELECT_DIR / "cl5_extra.parquet", index=False)
    return plan


class RoutingSimulator:
    """Classifier decisions at a miss weight, routing triggers and LLM fusion on cl5 / LOSO / LOCO.

    Kept documents are costed exactly. Routed documents with an LLM answer are costed exactly; routed documents
    without one are estimated: in LOSO / LOCO (uniform LLM samples per holdout) by scaling the holdout's answered
    routed documents, in cl5 (samples picked by the triggers, plus uniform samples in the wider bands) by the
    distribution of the fused label given the true label and the classifier's label among answered routed cl5
    documents of the same confidence band.
    """

    def __init__(self, spec: str, variant: str = "codebook", llm_result=LLM_RESULT):
        from .splits import cluster_folds, unique_doc_folds

        frame = holdout_frame(spec)
        keys = ["kind", "holdout", "doc_index"]
        frame = frame.merge(pd.read_parquet(NOVELTY_RESULT), on=keys, how="left")
        llm = pd.read_parquet(llm_result)
        columns = [f"p_{label}_{variant}" for label in LABELS]
        frame = frame.merge(llm[keys + [f"llm_{variant}"] + columns], on=keys, how="left")
        code = {label: i for i, label in enumerate(LABELS)}
        self.truth = frame["label"].map(code).to_numpy()
        self.proba = frame[[f"p_{label}" for label in LABELS]].to_numpy()
        self.answered = frame[f"llm_{variant}"].isin(LABELS).to_numpy()
        p_llm = frame[columns].to_numpy(dtype=float)
        # A reply without label probabilities counts as certain of its label.
        onehot = np.eye(3)[frame[f"llm_{variant}"].map(code).fillna(0).astype(int).to_numpy()]
        self.p_llm = np.where(np.isnan(p_llm).any(axis=1, keepdims=True), onehot, p_llm)
        self.maxsim = frame["maxsim"].to_numpy()
        self.holdout = HoldoutCosts(frame["kind"].to_numpy(), frame["holdout"].to_numpy(), self.truth)
        self.is_cl5 = (frame["kind"] == "cl5").to_numpy()
        self.loso = (frame["kind"] == "loso").to_numpy()
        unique = unique_documents(events("train"))
        fold_of = {"ud5": unique_doc_folds(unique), "cl5": cluster_folds(unique)}
        allowed = np.zeros((len(frame), 3), dtype=bool)
        sources = frame["source"].to_numpy()
        for g, (kind, name) in enumerate(self.holdout.names):
            cells = training_cells(unique, kind, name, fold_of)
            rows = np.flatnonzero(self.holdout.group == g)
            for i, label in enumerate(LABELS):
                allowed[rows, i] = [(src, label) in cells for src in sources[rows]]
        self.allowed = allowed
        self._decisions = {}

    def decision(self, m_dec: float):
        if m_dec not in self._decisions:
            costs = cost_matrix(m_dec)
            self._decisions[m_dec] = ((self.proba @ costs).argmin(axis=1), cost_confidence(self.proba, costs))
        return self._decisions[m_dec]

    def routed(self, m_dec: float, t: float, s: float, pair: bool, unseen: bool) -> np.ndarray:
        clf, conf = self.decision(m_dec)
        routed = conf < t
        if s > 0:
            routed |= self.maxsim < s
        if unseen:
            routed |= self.loso
        if pair:
            routed |= ~self.allowed[np.arange(len(clf)), clf]
        return routed

    def fuse(self, m_dec: float, rule: str, param: float) -> np.ndarray:
        """The label of every document if routed and answered (``rule`` "none" keeps the classifier's)."""

        clf, _ = self.decision(m_dec)
        if rule == "none":
            return clf
        return fuse_codes(rule, param, clf, self.proba, self.p_llm, m_dec)

    def confusion(self, m_dec: float, routed: np.ndarray, rule: str, param: float) -> np.ndarray:
        clf, _ = self.decision(m_dec)
        fused = self.fuse(m_dec, rule, param)
        hc = self.holdout
        known = routed & self.answered
        unknown = routed & ~self.answered
        confusion = hc.confusion(clf, (~routed).astype(float)) + hc.confusion(fused, known.astype(float))
        n_groups = len(hc.names)
        # LOSO / LOCO: scale each holdout's answered routed documents up to all of its routed documents.
        n_routed = np.bincount(hc.group, weights=routed.astype(float), minlength=n_groups)
        n_known = np.bincount(hc.group, weights=known.astype(float), minlength=n_groups)
        known_conf = hc.confusion(fused, known.astype(float))
        ood = hc.kind != "cl5"
        scale = np.where(n_known > 0, n_routed / np.maximum(n_known, 1), 0.0) - 1.0
        confusion[ood] += known_conf[ood] * scale[ood, None, None]
        # cl5, and LOSO / LOCO holdouts without a single answered routed document: P(fused | truth, clf) among
        # answered routed cl5 documents in the same confidence band (the LLM errs more on the least sure ones),
        # else over all bands, else P(fused | truth).
        _, conf = self.decision(m_dec)
        band = np.digitize(conf, CONFIDENCE_EDGES)
        stratified = np.zeros((len(CONFIDENCE_EDGES) + 1, 3, 3, 3))
        sample = known & self.is_cl5
        np.add.at(stratified, (band[sample], self.truth[sample], clf[sample], fused[sample]), 1.0)
        pooled = stratified.sum(axis=0)
        fallback = np.zeros((3, 3))
        np.add.at(fallback, (self.truth[known], fused[known]), 1.0)
        estimate = unknown & (self.is_cl5 | (n_known[hc.group] == 0))
        if estimate.any():
            t, c, b = self.truth[estimate], clf[estimate], band[estimate]
            dist = stratified[b, t, c]
            whole = pooled[t, c]
            fb = fallback[t] / np.maximum(fallback[t].sum(axis=1, keepdims=True), 1)
            dist = np.where(dist.sum(axis=1, keepdims=True) > 0, dist / np.maximum(dist.sum(axis=1, keepdims=True), 1),
                            np.where(whole.sum(axis=1, keepdims=True) > 0,
                                     whole / np.maximum(whole.sum(axis=1, keepdims=True), 1), fb))
            # Nothing answered at all: the classifier's label stands.
            none = dist.sum(axis=1) == 0
            dist[none, c[none]] = 1.0
            cell = hc.group[estimate] * 3 + t
            for j in range(3):
                confusion[:, :, j] += np.bincount(cell, weights=dist[:, j], minlength=n_groups * 3).reshape(n_groups, 3)
        return confusion


def routing(specs=STAGE2, max_routed: float = MAX_ROUTED) -> pd.DataFrame:
    rows = []
    for spec in specs:
        sim = RoutingSimulator(spec)
        cl5_docs = sim.holdout.docs[sim.holdout.kind == "cl5"].sum()
        for m_dec in ROUTE_MISS:
            configs = [("none", 0.0, 0.0, False, False, "none", 0.0)]
            for t, s, pair, unseen in itertools.product(DOUBT, NOVELTY, (False, True), (False, True)):
                if t == 0 and s == 0 and not pair and not unseen:
                    continue
                for rule, values in (("gate_km", LLM_MISS), ("raise", LLM_MISS), ("mix_km", MIX)):
                    configs += [(f"{rule}({value:g})", t, s, pair, unseen, rule, value) for value in values]
            for name, t, s, pair, unseen, rule, param in configs:
                routed = sim.routed(m_dec, t, s, pair, unseen) if rule != "none" else np.zeros(len(sim.truth), dtype=bool)
                confusion = sim.confusion(m_dec, routed, rule, param)
                row = {"classifier": spec, "m_dec": m_dec, "t": t, "s": s, "pair": pair, "unseen": unseen, "fusion": name,
                       "cl5_routed_share": routed[sim.is_cl5].sum() / cl5_docs,
                       "loco_routed_share": routed[sim.holdout.kind[sim.holdout.group] == "loco"].mean()}
                for miss in MISS_COSTS:
                    for key, value in sim.holdout.aggregate(confusion, miss).items():
                        row[f"m{miss:g}_{key}"] = value
                for kind in KINDS:
                    row.update(error_counts(sim.holdout, confusion, kind))
                rows.append(row)
        print(f"{spec}: {len(rows)} configurations so far")
    grid = pd.DataFrame(rows)
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(SELECT_DIR / "routing_grid.parquet", index=False)
    feasible = add_regret(grid[grid["cl5_routed_share"] <= max_routed].copy()).sort_values("regret")
    best = feasible.iloc[0].to_dict()
    without_llm = feasible[feasible["fusion"] == "none"].iloc[0].to_dict()
    per_miss = {f"m{m:g}": feasible.sort_values([f"regret_m{m:g}", "regret"]).iloc[0].to_dict() for m in MISS_COSTS}
    report = {"max_routed": max_routed, "best": best, "best_without_llm": without_llm, "best_per_miss": per_miss}
    (SELECT_DIR / "routing.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["classifier", "m_dec", "t", "s", "pair", "unseen", "fusion", "regret", *[f"regret_m{m:g}" for m in MISS_COSTS],
            "cl5_routed_share", "cl5_threat_miss_rate", "cl5_false_alarm_rate", "loso_threat_miss_rate", "loco_threat_miss_rate"]
    print(feasible[show].head(25).round(5).to_string(index=False))
    print("\nbest without LLM:", {k: without_llm[k] for k in show})
    for key, row in per_miss.items():
        print(f"best if {key}:", {k: row[k] for k in show})
    return feasible


ROBUST_MIN_DOCS = 30  # folds smaller than this are left out of the "macro30" aggregation


def aggregation_check() -> pd.DataFrame:
    """Does the choice survive aggregations that keep tiny folds from deciding it?

    Without the routing cap the minimax choice moved to doubt 0.99, but the gain came from a 12-document LOCO fold
    (Duo suspicious) weighted like the 136,729-document AD fold, and the choice changed with the candidate set. This
    re-scores a fixed candidate set (unseen-source routing on, no novelty trigger, the two competitive fusions) per
    fold and selects under four OOD aggregations: macro (every fold the same), macro over folds of at least
    ROBUST_MIN_DOCS documents, folds weighted by the square root of their size, and pooled.
    """

    aggregations = ("macro", "macro30", "sqrt", "pooled")
    rows = []
    for spec in STAGE2:
        sim = RoutingSimulator(spec)
        hc = sim.holdout
        for m_dec, t, pair in itertools.product(BAND_MISS, (0.7, 0.9, 0.95, 0.99), (False, True)):
            routed = sim.routed(m_dec, t, 0.0, pair, True)
            for rule, param in (("mix_km", 0.5), ("mix_km", 0.75), ("gate_km", 1), ("gate_km", 2)):
                confusion = sim.confusion(m_dec, routed, rule, param)
                row = {"classifier": spec, "m_dec": m_dec, "t": t, "pair": pair, "fusion": f"{rule}({param:g})",
                       "cl5_routed_share": routed[sim.is_cl5].mean()}
                for miss in MISS_COSTS:
                    cost = (confusion * cost_matrix(miss)).sum(axis=(1, 2))
                    per_doc = cost / hc.docs
                    cl5 = hc.kind == "cl5"
                    for agg in aggregations:
                        ood = []
                        for kind in ("loso", "loco"):
                            mask = hc.kind == kind
                            if agg == "macro30":
                                mask = mask & (hc.docs >= ROBUST_MIN_DOCS)
                            if agg in ("macro", "macro30"):
                                ood.append(per_doc[mask].mean())
                            elif agg == "sqrt":
                                weight = np.sqrt(hc.docs[mask])
                                ood.append((per_doc[mask] * weight).sum() / weight.sum())
                            else:
                                ood.append(cost[mask].sum() / hc.docs[mask].sum())
                        row[f"m{miss:g}_{agg}"] = ((1 - PRIOR_OOD) * cost[cl5].sum() / hc.docs[cl5].sum()
                                                   + PRIOR_OOD * np.mean(ood))
                rows.append(row)
    table = pd.DataFrame(rows)
    chosen = (table["classifier"] == STAGE2[1]) & (table["m_dec"] == 2) & (table["t"] == 0.9) & table["pair"] \
        & (table["fusion"] == "mix_km(0.75)")
    report = {}
    for criterion in (("macro", "pooled"), ("macro30", "pooled"), ("sqrt", "pooled"), ("pooled",), ("macro30",), ("sqrt",)):
        columns = [f"m{m:g}_{agg}" for m in MISS_COSTS for agg in criterion]
        regret = np.max(np.vstack([table[c] / table[c].min() for c in columns]), axis=0)
        best = table.assign(regret=regret).sort_values("regret").iloc[0]
        name = "+".join(criterion)
        report[name] = {**{k: best[k] for k in ("classifier", "m_dec", "t", "pair", "fusion", "cl5_routed_share", "regret")},
                        "regret_of_cost1_triggers": float(regret[chosen.to_numpy()][0])}
        print(f"{name:16s} best: {best['classifier']} m_dec {best['m_dec']} t {best['t']} pair {best['pair']} "
              f"{best['fusion']} (cl5 routed {best['cl5_routed_share']:.3f}, regret {best['regret']:.3f}); "
              f"cost1 triggers {report[name]['regret_of_cost1_triggers']:.3f}")
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SELECT_DIR / "robust.csv", index=False)
    (SELECT_DIR / "robust.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["classifiers", "picks", "routing", "robust"])
    parser.add_argument("--max-routed", type=float, default=MAX_ROUTED)
    args = parser.parse_args()
    if args.stage == "classifiers":
        classifiers()
    elif args.stage == "picks":
        picks()
    elif args.stage == "robust":
        aggregation_check()
    else:
        routing(max_routed=args.max_routed)


if __name__ == "__main__":
    main()
