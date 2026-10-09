"""Document novelty: cosine similarity to the nearest training document in an LSA space.

The classifier's own confidence does not flag distribution shift: on the
leave-one-source-out holdouts its errors have a median confidence above 0.98
(unseen Crowdstrike detections are called benign at 0.9998). A nearest-neighbour
distance does: held-out documents of known sources sit at similarity ~1.0, an
unseen source at 0.56-0.71.

python -m soc_hybrid.novelty  → artifacts/hybrid/novelty_holdout.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import OUT, events, unique_documents
from .holdout import cap_cells
from .models import MAX_DOC_CHARS
from .splits import holdouts

RESULT = OUT / "novelty_holdout.parquet"


class Novelty:
    def __init__(self, dims: int = 200, seed: int = 0):
        self.dims, self.seed = dims, seed

    def fit(self, docs):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]+\b|0", ngram_range=(1, 2),
                                          max_features=200_000, min_df=2, sublinear_tf=True, dtype=np.float32)
        x = self.vectorizer.fit_transform([doc[:MAX_DOC_CHARS] for doc in docs])
        self.svd = TruncatedSVD(self.dims, random_state=self.seed).fit(x)
        self.reference = normalize(self.svd.transform(x)).astype(np.float32)
        return self

    def max_similarity(self, docs, chunk: int = 4096) -> np.ndarray:
        x = self.vectorizer.transform([doc[:MAX_DOC_CHARS] for doc in docs])
        query = normalize(self.svd.transform(x)).astype(np.float32)
        best = np.empty(len(query), dtype=np.float32)
        for start in range(0, len(query), chunk):
            best[start : start + chunk] = (query[start : start + chunk] @ self.reference.T).max(axis=1)
        return best


def _fold(unique: pd.DataFrame, mask: np.ndarray, cap: int) -> np.ndarray:
    train = cap_cells(unique[~mask], cap)
    return Novelty().fit(train["doc"].tolist()).max_similarity(unique.loc[mask, "doc"].tolist())


def main() -> None:
    unique = unique_documents(events("train"))
    plan = holdouts(unique)
    scores = Parallel(n_jobs=8, verbose=5)(delayed(_fold)(unique, mask, 20_000) for _, _, mask in plan)
    parts = [pd.DataFrame({"kind": kind, "holdout": name, "doc_index": np.flatnonzero(mask), "maxsim": score})
             for (kind, name, mask), score in zip(plan, scores)]
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(RESULT, index=False)
    print(result.groupby("kind")["maxsim"].describe(percentiles=[0.01, 0.05, 0.5]).round(3).to_string())


if __name__ == "__main__":
    main()
