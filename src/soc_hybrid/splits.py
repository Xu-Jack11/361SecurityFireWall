"""Train-internal holdouts that imitate the shifts a deployment can meet.

All three work on unique (doc, label) rows of train only:

- unique-document 5-fold: unseen documents of known sources and record types;
- leave one source out (LOSO): a log source the classifier never saw;
- leave one (source, label) cell out (LOCO): a known source sending a record
  type, and label, the classifier never saw from it. This is the case where a
  classifier falls back on "this source means that label";
- cluster 5-fold (cl5, iteration 2): unique-document folds put near-duplicates
  on both sides (90% of held-out documents sit at similarity >= 0.994 to a
  training document, against a median of 0.985 for valid), so documents are
  grouped by KMeans clusters of their LSA vectors and whole clusters are held out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def unique_doc_folds(unique: pd.DataFrame, k: int = 5, seed: int = 0) -> np.ndarray:
    folds = np.zeros(len(unique), dtype=int)
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(unique)), unique["label"])):
        folds[test] = fold
    return folds


CLUSTERS = 300


def document_clusters(unique: pd.DataFrame, clusters: int = CLUSTERS, seed: int = 0) -> np.ndarray:
    """KMeans cluster of every unique document's LSA vector (LSA fitted on capped train cells; cached)."""

    from pathlib import Path

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    from .data import CACHE

    path = Path(CACHE) / f"clusters_k{clusters}_s{seed}.npy"
    if path.exists():
        return np.load(path)
    from .holdout import cap_cells
    from .novelty import Novelty

    lsa = Novelty(seed=seed).fit(cap_cells(unique, 20_000)["doc"].tolist())
    x = lsa.vectorizer.transform([doc[:6000] for doc in unique["doc"]])
    vectors = normalize(lsa.svd.transform(x)).astype(np.float32)
    labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed, batch_size=8192, n_init=3).fit_predict(vectors)
    np.save(path, labels)
    return labels


def cluster_folds(unique: pd.DataFrame, k: int = 5, clusters: int = CLUSTERS, seed: int = 0) -> np.ndarray:
    groups = document_clusters(unique, clusters, seed)
    folds = np.zeros(len(unique), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(unique)), unique["label"], groups)):
        folds[test] = fold
    return folds


def holdouts(unique: pd.DataFrame, k: int = 5, seed: int = 0, clusters: int = CLUSTERS) -> list[tuple[str, str, np.ndarray]]:
    """(kind, name, test mask) for every train-internal holdout."""

    result = []
    folds = unique_doc_folds(unique, k, seed)
    for fold in range(k):
        result.append(("ud5", f"fold{fold}", folds == fold))
    folds = cluster_folds(unique, k, clusters, seed)
    for fold in range(k):
        result.append(("cl5", f"fold{fold}", folds == fold))
    for source in sorted(unique["source"].unique()):
        result.append(("loso", source, (unique["source"] == source).to_numpy()))
    labels_per_source = unique.groupby("source")["label"].nunique()
    for source in sorted(labels_per_source[labels_per_source > 1].index):
        for label in sorted(unique.loc[unique["source"] == source, "label"].unique()):
            mask = ((unique["source"] == source) & (unique["label"] == label)).to_numpy()
            result.append(("loco", f"{source}|{label}", mask))
    return result
