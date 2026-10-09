"""Decisions over class probabilities: benign-first (the original rule) and minimum expected cost (the corrected one)."""

from __future__ import annotations

import numpy as np

from .data import LABELS

SWAP_COST = 0.5  # suspicious and malicious called each other earn partial credit


def cost_matrix(miss: float, swap: float = SWAP_COST, false_alarm: float = 1.0) -> np.ndarray:
    """Rows = true label, columns = predicted label, both in LABELS order (benign, suspicious, malicious).

    A threat called benign costs ``miss``, a benign event raised as an alert costs ``false_alarm`` and a
    suspicious/malicious swap costs ``swap``.
    """

    return np.array([[0.0, false_alarm, false_alarm], [miss, 0.0, swap], [miss, swap, 0.0]])


def min_cost(proba: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """The label with the smallest expected cost under ``costs`` (the Bayes decision for calibrated proba)."""

    risk = np.asarray(proba, dtype=float) @ costs
    return np.asarray(LABELS, dtype=object)[risk.argmin(axis=1)]


def fuse_codes(rule: str, param: float, clf: np.ndarray, proba: np.ndarray, p_llm: np.ndarray, m_dec: float) -> np.ndarray:
    """Label codes (0 benign, 1 suspicious, 2 malicious) after the LLM, for documents it answered.

    - ``gate_km``: the LLM decides benign or alert by minimum cost at miss weight ``param``; the classifier keeps
      its malicious decisions (the LLM does not know this dataset's malicious class) and picks the alert type.
    - ``raise``: the LLM can only raise an alert, at miss weight ``param``, on what the classifier calls benign.
    - ``mix_km``: the two distributions averaged with LLM share ``param`` decide benign or alert at the
      classifier's miss weight ``m_dec``; malicious kept and alert type as for ``gate_km``.
    """

    clf = np.asarray(clf)
    alert_type = 1 + np.asarray(proba)[:, 1:].argmax(axis=1)
    if rule == "gate_km":
        llm_alert = (np.asarray(p_llm) @ cost_matrix(param)).argmin(axis=1) != 0
        return np.where(clf == 2, 2, np.where(llm_alert, alert_type, 0))
    if rule == "raise":
        llm_alert = (np.asarray(p_llm) @ cost_matrix(param)).argmin(axis=1) != 0
        return np.where((clf == 0) & llm_alert, alert_type, clf)
    if rule == "mix_km":
        mixed = (1 - param) * np.asarray(proba) + param * np.asarray(p_llm)
        decided = (mixed @ cost_matrix(m_dec)).argmin(axis=1)
        return np.where(clf == 2, 2, np.where(decided == 0, 0, alert_type))
    raise ValueError(rule)


def cost_confidence(proba: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """How far the minimum-cost decision is from flipping: r2 / (r1 + r2) for the two smallest expected costs.

    1 = no doubt (the chosen label costs nothing), 0.5 = a tie, as for ``decision_confidence``.
    """

    risk = np.sort(np.asarray(proba, dtype=float) @ costs, axis=1)
    total = risk[:, 0] + risk[:, 1]
    return np.where(total > 0, risk[:, 1] / np.where(total > 0, total, 1.0), 1.0)


def benign_first(proba: np.ndarray, benign_weight: float) -> np.ndarray:
    """Argmax of p(y|x) * weight[y] with weight[benign] = w and 1 for the alerts.

    An alert is raised only when it is more than w times as likely as benign:
    with a benign miss costing w times another error this is the Bayes decision.
    """

    scores = np.asarray(proba, dtype=float).copy()
    scores[:, 0] *= benign_weight
    return np.asarray(LABELS, dtype=object)[scores.argmax(axis=1)]


def decision_confidence(proba: np.ndarray, benign_weight: float) -> np.ndarray:
    """How far the benign-first decision is from flipping, as a probability-like margin in [0, 1].

    The share of the decided label's weighted score among all weighted scores:
    1 = no doubt, 0.5 = a tie between two labels.
    """

    scores = np.asarray(proba, dtype=float).copy()
    scores[:, 0] *= benign_weight
    return scores.max(axis=1) / scores.sum(axis=1)
