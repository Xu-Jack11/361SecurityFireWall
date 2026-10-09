"""The LLM on train-internal holdouts: how well does it label sources and record types it was not told about?

For every LOSO / LOCO holdout of splits.py, up to --per-holdout held-out unique
documents are sampled and sent to the LLM with the codebook minus every line
read only off the held-out cells (and, for the ablation, with the generic
definitions). ud5 documents a routing trigger can pick are added by ``--from-runs``
once classifier holdout runs and novelty scores exist.

python -m soc_hybrid.llm_holdout [--per-holdout 40] [--variants codebook,generic] [--from-runs RUN,... --ud5-docs 300]
  → artifacts/hybrid/llm_holdout.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import LABELS, OUT, events, unique_documents
from .decision import benign_first, decision_confidence
from .holdout import HOLDOUT_DIR
from .llm import LLMRunner, requests_for
from .splits import holdouts

RESULT = OUT / "llm_holdout.parquet"


def excluded_cells(kind: str, name: str, unique: pd.DataFrame) -> set:
    if kind == "loso":
        return {(name, label) for label in LABELS}
    if kind == "loco":
        source, label = name.rsplit("|", 1)
        return {(source, label)}
    return set()


def sample_plan(unique: pd.DataFrame, per_holdout: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for kind, name, mask in holdouts(unique):
        if kind == "ud5":
            continue
        index = np.flatnonzero(mask)
        chosen = rng.choice(index, size=min(per_holdout, len(index)), replace=False)
        parts.append(pd.DataFrame({"kind": kind, "holdout": name, "doc_index": chosen, "holdout_docs": len(index)}))
    return pd.concat(parts, ignore_index=True)


def indist_plan(unique: pd.DataFrame, runs: list[str], docs: int, kind: str = "ud5",
                weights=(1, 10)) -> pd.DataFrame:
    """In-distribution documents a routing trigger can pick: least sure (per benign weight), least similar,
    new (source, label) pairs."""

    from .novelty import RESULT as NOVELTY_RESULT
    from .route_select import training_cells
    from .splits import cluster_folds, unique_doc_folds

    picked = []
    fold_of = {"ud5": unique_doc_folds(unique), "cl5": cluster_folds(unique)}
    for run in runs:
        probs = pd.read_parquet(HOLDOUT_DIR / run / "probs.parquet")
        part = probs[probs["kind"] == kind].copy()
        proba = part[[f"p_{label}" for label in LABELS]].to_numpy()
        cells = {h: training_cells(unique, kind, h, fold_of) for h in part["holdout"].unique()}
        for w in weights:
            part["conf"] = decision_confidence(proba, w)
            picked.append(part.nsmallest(docs, "conf")[["holdout", "doc_index"]])
            part["clf"] = benign_first(proba, w)
            new = [(src, lab) not in cells[h] for h, src, lab in zip(part["holdout"], part["source"], part["clf"])]
            picked.append(part.loc[new, ["holdout", "doc_index"]].head(docs))
    novelty = pd.read_parquet(NOVELTY_RESULT)
    picked.append(novelty[novelty["kind"] == kind].nsmallest(docs, "maxsim")[["holdout", "doc_index"]])
    plan = pd.concat(picked).drop_duplicates()
    return plan.assign(kind=kind, holdout_docs=len(unique))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-holdout", type=int, default=40)
    parser.add_argument("--variants", default="codebook,generic")
    parser.add_argument("--from-runs", default=None, help="classifier holdout runs whose trigger-picked ud5 docs to add")
    parser.add_argument("--ud5-docs", type=int, default=300)
    parser.add_argument("--model", default="qwen3.5-4b", choices=["qwen3.5-4b", "qwen3-4b-2507"])
    parser.add_argument("--ud5-extra", default=None, help="parquet of (holdout, doc_index) ud5 documents to add")
    parser.add_argument("--cl5-runs", default=None, help="classifier runs whose trigger-picked cl5 documents to add")
    parser.add_argument("--cl5-docs", type=int, default=100)
    parser.add_argument("--cl5-extra", default=None, help="parquet of (holdout, doc_index) cl5 documents to add")
    parser.add_argument("--keep-previous", action="store_true",
                        help="keep every document already in the result (their cached replies are reused)")
    args = parser.parse_args()
    tag = "" if args.model == "qwen3.5-4b" else f"__{args.model}"

    unique = unique_documents(events("train"))
    plan = sample_plan(unique, args.per_holdout)
    if args.keep_previous and RESULT.exists():
        plan = pd.concat([plan, pd.read_parquet(RESULT, columns=["kind", "holdout", "doc_index", "holdout_docs"])],
                         ignore_index=True)
    if args.cl5_extra:
        extra = pd.read_parquet(args.cl5_extra).assign(kind="cl5", holdout_docs=len(unique))
        plan = pd.concat([plan, extra], ignore_index=True)
    if args.from_runs:
        plan = pd.concat([plan, indist_plan(unique, args.from_runs.split(","), args.ud5_docs)], ignore_index=True)
    if args.cl5_runs:
        plan = pd.concat([plan, indist_plan(unique, args.cl5_runs.split(","), args.cl5_docs, "cl5",
                                            weights=(1, 3, 10, 30, 100, 300))], ignore_index=True)
    if args.ud5_extra:
        extra = pd.read_parquet(args.ud5_extra).assign(kind="ud5", holdout_docs=len(unique))
        plan = pd.concat([plan, extra], ignore_index=True)
    plan = plan.drop_duplicates(["kind", "holdout", "doc_index"]).reset_index(drop=True)
    docs = unique.iloc[plan["doc_index"].to_numpy()].reset_index(drop=True)
    result = plan.assign(label=docs["label"].to_numpy(), source=docs["source"].to_numpy(), rows=docs["rows"].to_numpy())

    runner = LLMRunner("holdout", model=args.model)
    for variant in args.variants.split(","):
        requests = []
        for (kind, name), group in result.groupby(["kind", "holdout"], sort=False):
            excluded = excluded_cells(kind, name, unique)
            for position, request in zip(group.index, requests_for(docs.loc[group.index], variant, excluded)):
                requests.append((position, request))
        requests.sort()
        replies = runner.run([request for _, request in requests])
        result[f"llm_{variant}{tag}"] = [reply["label"] for reply in replies]
        for label in LABELS:
            # Label probability at the JSON label token; a reply without one counts as certain of its label.
            result[f"p_{label}_{variant}{tag}"] = [
                (reply["probs"] or {}).get(label, float(reply["label"] == label)) if reply.get("probs") is not None
                else float(reply["label"] == label) for reply in replies]
        result[f"tokens_{variant}{tag}"] = [reply["prompt_tokens"] + reply["output_tokens"] for reply in replies]
    if RESULT.exists():
        # Keep the other models' and variants' columns for the documents both plans share.
        previous = pd.read_parquet(RESULT)
        keys = ["kind", "holdout", "doc_index"]
        extra = [c for c in previous.columns if c.startswith(("llm_", "tokens_", "p_")) and c not in result.columns]
        result = result.merge(previous[keys + extra], on=keys, how="left")
    result.to_parquet(RESULT, index=False)
    for variant in args.variants.split(","):
        column = f"llm_{variant}{tag}"
        hit = result[column] == result["label"]
        alert_hit = (result[column] != "benign") == (result["label"] != "benign")
        print(column, "accuracy", round(float(hit.mean()), 3), "alert accuracy", round(float(alert_hit.mean()), 3),
              "unparsed", int(result[column].isna().sum()))
        print(result.assign(hit=hit).groupby(["kind", "holdout"])["hit"].agg(["mean", "size"]).round(2).to_string())


if __name__ == "__main__":
    main()
