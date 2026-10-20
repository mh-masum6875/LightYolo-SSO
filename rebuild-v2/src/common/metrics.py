"""WP3 metrics module (Master Plan 7): accuracy, F1, AUROC, and false alarms
per node per day at 95 percent detection. Threshold swept on VALIDATION,
evaluated ONCE on test. Pure numpy (no sklearn dependency).

IBRL sampling: one reading per ~31 s epoch -> a 64-sample window spans
~33 min -> WINDOWS_PER_DAY ~ 43.4 per node.
"""
from __future__ import annotations
import numpy as np

IBRL_EPOCH_S = 31.0
W_LEN = 64
WINDOWS_PER_DAY = 24 * 3600 / (IBRL_EPOCH_S * W_LEN)   # ~43.4

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def f1(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = float((y_true & y_pred).sum())
    fp = float((~y_true & y_pred).sum())
    fn = float((y_true & ~y_pred).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0

def auroc(y_true, scores) -> float:
    """Mann-Whitney U formulation with tie handling (average ranks)."""
    y_true, scores = np.asarray(y_true).astype(bool), np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[y_true].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def threshold_at_detection(y_val, s_val, detection: float = 0.95) -> float:
    """Lowest threshold achieving >= `detection` recall on validation.
    Predict anomalous iff score >= threshold."""
    y_val, s_val = np.asarray(y_val).astype(bool), np.asarray(s_val, dtype=float)
    pos = np.sort(s_val[y_val])
    if len(pos) == 0:
        return float("inf")
    k = int(np.floor((1 - detection) * len(pos)))
    return float(pos[k])

def false_alarms_per_node_day(y_true, y_pred, n_nodes: int) -> float:
    """False positives normalised to alarms per node per day."""
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    fp = float((~y_true & y_pred).sum())
    n_windows = len(y_true)
    days = n_windows / (WINDOWS_PER_DAY * n_nodes)
    return fp / (n_nodes * days) if days > 0 else float("nan")

def evaluate_scores(y_val, s_val, y_test, s_test, n_nodes: int, detection: float = 0.95) -> dict:
    """The full WP3 protocol in one call: sweep on val, evaluate once on test."""
    thr = threshold_at_detection(y_val, s_val, detection)
    y_pred = np.asarray(s_test, dtype=float) >= thr
    return {
        "threshold": thr,
        "accuracy": accuracy(y_test, y_pred),
        "f1": f1(y_test, y_pred),
        "auroc": auroc(y_test, s_test),
        "false_alarms_per_node_day": false_alarms_per_node_day(y_test, y_pred, n_nodes),
        "recall_test": float((np.asarray(y_test).astype(bool) & y_pred).sum()
                             / max(1, int(np.asarray(y_test).astype(bool).sum()))),
    }
