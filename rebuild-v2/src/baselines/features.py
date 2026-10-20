"""12 window statistics for the classical baselines (Master Plan 7: ocsvm, iforest).
Per modality (4): std, range, lag-1 autocorrelation -> 12 features total.
Pure numpy so the module is testable without sklearn/torch.
"""
from __future__ import annotations
import numpy as np

def window_features(win: np.ndarray) -> np.ndarray:
    """win: (T, 4) -> (12,) [std_m0..3, range_m0..3, lag1ac_m0..3]"""
    stds = win.std(axis=0)
    rngs = win.max(axis=0) - win.min(axis=0)
    acs = np.empty(win.shape[1])
    for m in range(win.shape[1]):
        x = win[:, m]
        x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
        d = np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
        acs[m] = float((x0 * x1).sum() / d) if d > 1e-12 else 0.0
    return np.concatenate([stds, rngs, acs])

def batch_features(wins: np.ndarray) -> np.ndarray:
    return np.stack([window_features(w) for w in wins])
