"""End to end: fit the classifier on train, route valid events, ask the LLM, write valid predictions.

python -m soc_hybrid.pipeline --name cost1 --classifier textcnn --weights rows --params '{"device": "cuda"}' \
    --blend tfidf_word_lr:rows --miss-weight 2 --doubt 0.9 --pair --unseen-source \
    --fusion mix_km --fusion-param 0.75 --budget 2000 --priority doubt
  → artifacts/hybrid/runs/<name>/{config.json, docs.parquet, valid_pred.csv}

(the corrected scoring rule, I17; I16's runs used --benign-weight with the llm_* fusions instead)

Nothing here reads valid labels: docs.parquet holds every component (classifier
probabilities, routing, LLM label, final label) so evaluate.py can score the
system and its ablations afterwards.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from .data import LABELS, OUT, events, unique_documents
from .decision import benign_first, cost_confidence, cost_matrix, decision_confidence, fuse_codes, min_cost
from .holdout import cap_cells
from .llm import LLMRunner, requests_for
from .models import make_classifier, training_weights
from .route_select import fuse

RUNS = OUT / "runs"
MODELS = OUT / "models"
COST_FUSIONS = ("gate_km", "raise", "mix_km")  # corrected rule, see decision.fuse_codes


def model_key(classifier: str, weights: str, cap: int, params: dict) -> str:
    """Cache name of a fitted classifier; a stable digest of its parameters (built-in hash() is salted per process)."""

    import hashlib

    digest = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8] if params else ""
    return f"{classifier}_{weights}_cap{cap}" + (f"_{digest}" if digest else "")


def fit_classifier(classifier: str, weights: str, cap: int, params: dict, name: str):
    """Fit on all of train (cells capped as in the holdouts) and cache the model by its settings."""

    path = MODELS / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    train = cap_cells(unique_documents(events("train")), cap)
    model = make_classifier(classifier, **params).fit(train["doc"].tolist(), train["label"].to_numpy(),
                                                       training_weights(train, weights))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def fit_novelty(cap: int):
    path = MODELS / f"novelty_cap{cap}.joblib"
    if path.exists():
        return joblib.load(path)
    from .novelty import Novelty

    train = cap_cells(unique_documents(events("train")), cap)
    model = Novelty().fit(train["doc"].tolist())
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def valid_documents() -> pd.DataFrame:
    """One row per distinct valid document (classifier doc + LLM request fields), with its row count."""

    va = events("valid")
    keys = ["doc", "pipeline", "vendor_name", "product_name", "llm_message"]
    docs = va.groupby(keys, sort=False).agg(rows=("event_id", "size"), source=("source", "first")).reset_index()
    return docs


def triggers(docs: pd.DataFrame, doubt: float, novelty: float, pair: bool, unseen_source: bool,
             train_cells: set) -> pd.DataFrame:
    """Which routing triggers fire for each document (see route_select.py)."""

    train_sources = {source for source, _ in train_cells}
    fired = pd.DataFrame(index=docs.index)
    fired["unseen_source"] = unseen_source & ~docs["source"].isin(train_sources)
    fired["novelty"] = docs["maxsim"] < novelty if novelty > 0 else False
    fired["new_pair"] = pair & np.array([(s, l) not in train_cells for s, l in zip(docs["source"], docs["clf"])])
    fired["doubt"] = docs["conf"] < doubt
    return fired


def route(docs: pd.DataFrame, fired: pd.DataFrame, budget: int, random_seed: int | None = None,
          priority: str = "novelty") -> np.ndarray:
    """Documents sent to the LLM: unseen sources first, then the least similar (or, with
    ``priority="doubt"``, the least sure) first; at most ``budget`` (0 = every document a trigger picks)."""

    candidates = np.flatnonzero(fired.any(axis=1).to_numpy())
    if budget <= 0:
        budget = len(candidates)
    if random_seed is not None:
        # Ablation: as many documents as the triggers pick, drawn at random from all of valid.
        rng = np.random.default_rng(random_seed)
        order = rng.choice(len(docs), size=min(len(candidates), budget), replace=False)
    else:
        first, second = (docs["maxsim"], docs["conf"]) if priority == "novelty" else (docs["conf"], docs["maxsim"])
        key = np.lexsort((second.to_numpy()[candidates], first.to_numpy()[candidates],
                          ~fired["unseen_source"].to_numpy()[candidates]))
        order = candidates[key][:budget]
    mask = np.zeros(len(docs), dtype=bool)
    mask[order] = True
    return mask


def cost_fusion(docs: pd.DataFrame, proba: np.ndarray, llm_proba: np.ndarray, rule: str, param: float,
                miss_weight: float) -> np.ndarray:
    """Final labels under the corrected rule: routed documents the LLM answered go through ``fuse_codes``."""

    code = {label: i for i, label in enumerate(LABELS)}
    clf = docs["clf"].map(code).to_numpy()
    answered = docs["routed"].to_numpy(dtype=bool) & docs["llm"].isin(LABELS).to_numpy()
    q = llm_proba[answered]
    # A reply without label probabilities counts as certain of its label.
    onehot = np.eye(3)[docs.loc[answered, "llm"].map(code).to_numpy()]
    q = np.where(np.isnan(q).any(axis=1, keepdims=True), onehot, q)
    final = clf.copy()
    final[answered] = fuse_codes(rule, param, clf[answered], proba[answered], q, miss_weight)
    return np.asarray(LABELS, dtype=object)[final]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--classifier", default="tfidf_word_lr")
    parser.add_argument("--weights", default="rows", choices=["doc", "log", "rows"])
    parser.add_argument("--cap", type=int, default=20_000)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--blend", default=None,
                        help="second classifier averaged with the first, as CLASSIFIER:WEIGHTS[:JSON params]")
    parser.add_argument("--benign-weight", type=float, default=10.0)
    parser.add_argument("--miss-weight", type=float, default=None,
                        help="corrected rule: decide by minimum expected cost with this threat-miss weight "
                             "(replaces --benign-weight)")
    parser.add_argument("--doubt", type=float, default=0.0, help="route when the decision confidence is below this")
    parser.add_argument("--novelty", type=float, default=0.0, help="route when the nearest-train similarity is below this")
    parser.add_argument("--pair", action="store_true", help="route when the predicted label was never seen with the source")
    parser.add_argument("--unseen-source", action="store_true", help="route every source absent from train")
    parser.add_argument("--budget", type=int, default=3000,
                        help="max distinct documents sent to the LLM; 0 = no cap, every triggered document")
    parser.add_argument("--model", default="qwen3.5-4b", choices=["qwen3.5-4b", "qwen3-4b-2507"])
    parser.add_argument("--fusion", default="llm_prob", choices=["llm", "llm_gate", "agree_alert", "llm_prob", "llm_prob_gate",
                                                                  "llm_prob_gate_km", *COST_FUSIONS])
    parser.add_argument("--llm-weight", type=float, default=10.0, help="benign weight on the LLM's label probabilities")
    parser.add_argument("--fusion-param", type=float, default=10.0,
                        help="gate_km / raise: the LLM's miss weight; mix_km: the LLM's share of the average")
    parser.add_argument("--variant", default="codebook", choices=["codebook", "generic"])
    parser.add_argument("--random-routing", type=int, default=None, help="ablation: seed for random routing")
    parser.add_argument("--priority", default="novelty", choices=["novelty", "doubt"], help="order of routed documents")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    if args.fusion in COST_FUSIONS and args.miss_weight is None:
        parser.error(f"--fusion {args.fusion} needs --miss-weight")
    began = time.time()
    params = json.loads(args.params)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    model_name = model_key(args.classifier, args.weights, args.cap, params)
    model = fit_classifier(args.classifier, args.weights, args.cap, params, model_name)
    members = [model]
    docs = valid_documents()
    proba = model.predict_proba(docs["doc"].tolist())
    if args.blend:
        # Equal-weight average of two classifiers' probabilities (iteration 4: TextCNN + TF-IDF).
        classifier, weights, *rest = args.blend.split(":", 2)
        blend_params = json.loads(rest[0]) if rest else {}
        blend_name = model_key(classifier, weights, args.cap, blend_params)
        other = fit_classifier(classifier, weights, args.cap, blend_params, blend_name)
        members.append(other)
        proba = (proba + other.predict_proba(docs["doc"].tolist())) / 2
        model_name = f"{model_name}+{blend_name}"
    for member in members:
        if getattr(member, "net", None) is not None:  # free the GPU for vLLM
            import torch

            member.net.to("cpu")
            torch.cuda.empty_cache()
    for i, label in enumerate(LABELS):
        docs[f"p_{label}"] = proba[:, i]
    if args.miss_weight is not None:
        docs["clf"] = min_cost(proba, cost_matrix(args.miss_weight))
        docs["conf"] = cost_confidence(proba, cost_matrix(args.miss_weight))
    else:
        docs["clf"] = benign_first(proba, args.benign_weight)
        docs["conf"] = decision_confidence(proba, args.benign_weight)
    docs["maxsim"] = fit_novelty(args.cap).max_similarity(docs["doc"].tolist())
    train = unique_documents(events("train"))
    fired = triggers(docs, args.doubt, args.novelty, args.pair, args.unseen_source, set(zip(train["source"], train["label"])))
    for name in fired.columns:
        docs[f"trigger_{name}"] = fired[name].to_numpy()
    docs["routed"] = False if args.no_llm else route(docs, fired, args.budget, args.random_routing, args.priority)
    docs["llm"] = None
    for label in LABELS:
        docs[f"llm_p_{label}"] = np.nan
    if docs["routed"].any():
        routed = docs[docs["routed"]]
        replies = LLMRunner("valid", model=args.model).run(requests_for(routed, args.variant))
        docs.loc[routed.index, "llm"] = [reply["label"] for reply in replies]
        for label in LABELS:
            docs.loc[routed.index, f"llm_p_{label}"] = [
                (reply["probs"] or {}).get(label) if reply.get("probs") else float(reply["label"] == label) for reply in replies]
    llm_proba = docs[[f"llm_p_{label}" for label in LABELS]].to_numpy(dtype=float)
    if args.fusion in COST_FUSIONS:
        docs["final"] = cost_fusion(docs, proba, llm_proba, args.fusion, args.fusion_param, args.miss_weight)
    else:
        docs["final"] = [fuse(args.fusion, l, c, p, None if np.isnan(q).any() else q, args.llm_weight) if r else c
                         for l, c, p, q, r in zip(docs["llm"], docs["clf"], proba, llm_proba, docs["routed"])]
    docs.drop(columns=["doc", "llm_message"]).to_parquet(out / "docs.parquet", index=False)

    va = events("valid")
    final = va[["event_id", "doc", "pipeline", "vendor_name", "product_name", "llm_message"]].merge(
        docs[["doc", "pipeline", "vendor_name", "product_name", "llm_message", "final"]],
        on=["doc", "pipeline", "vendor_name", "product_name", "llm_message"], how="left")
    submission = pd.DataFrame({"event_id": final["event_id"], "pred_label": final["final"]})
    assert len(submission) == len(va) and submission["event_id"].is_unique
    assert submission["pred_label"].isin(LABELS).all()
    submission.to_csv(out / "valid_pred.csv", index=False)
    config = {**vars(args), "model": model_name, "valid_docs": int(len(docs)), "routed_docs": int(docs["routed"].sum()),
              "routed_rows": int(docs.loc[docs["routed"], "rows"].sum()), "seconds": round(time.time() - began)}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
