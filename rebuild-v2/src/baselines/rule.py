"""The original stability threshold rule, reported as the floor baseline
(Master Plan 7). A window is NORMAL iff, for every modality on the 0-255
scale, std <= 10 percent (of 255) AND range <= 30 percent. These are the
exact thresholds that labelled the original curated dataset, which is why
the rule scores perfectly on the continuity task by construction; the paper
states this openly and a test asserts it.
"""
from __future__ import annotations
import numpy as np

STD_MAX = 0.10 * 255.0
RANGE_MAX = 0.30 * 255.0

def rule_is_normal(win: np.ndarray) -> bool:
    stds = win.std(axis=0)
    rngs = win.max(axis=0) - win.min(axis=0)
    return bool(np.all(stds <= STD_MAX) and np.all(rngs <= RANGE_MAX))

def rule_scores(wins: np.ndarray) -> np.ndarray:
    """Anomaly score in {0,1}: 1 = anomalous (rule violated)."""
    return np.array([0.0 if rule_is_normal(w) else 1.0 for w in wins])
