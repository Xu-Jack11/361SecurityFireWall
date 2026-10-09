"""Ablation: the LLM alone, on a uniform sample of distinct valid documents.

python -m soc_hybrid.llm_only [--docs 500]  → artifacts/hybrid/runs/llm_only_sample/docs.parquet
Scored by evaluate-style code in report.py; nothing here reads valid labels.
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import LABELS
from .llm import LLMRunner, requests_for
from .pipeline import RUNS, valid_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=500)
    args = parser.parse_args()
    docs = valid_documents()
    sample = docs.sample(n=args.docs, random_state=0)
    replies = LLMRunner("valid").run(requests_for(sample, "codebook"))
    sample = sample.assign(llm=[r["label"] for r in replies])
    for label in LABELS:
        sample[f"llm_p_{label}"] = [(r["probs"] or {}).get(label, np.nan) if r.get("probs") else float(r["label"] == label)
                                    for r in replies]
    out = RUNS / "llm_only_sample"
    out.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out / "docs.parquet", index=False)
    print(sample["llm"].value_counts().to_dict())


if __name__ == "__main__":
    main()
