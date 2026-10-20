"""WP2 shared search space: the exact 15-D bounds and discretisation of the
legacy Stage 1 architecture search (sso_architecture_optimizer.py, WP0 audit).
All strategies operate on the same continuous box and the same discretiser,
so evaluated candidates are directly comparable across methods.
"""
from __future__ import annotations
import json
import numpy as np

# [min, max] per dimension, verbatim from legacy _define_search_space()
BOUNDS = np.array([
    [16, 64],      # 0  initial channels            (int)
    [0.25, 1.0],   # 1  width multiplier P3
    [0.25, 1.0],   # 2  width multiplier P4
    [0.25, 1.0],   # 3  width multiplier P5
    [0.5, 2.0],    # 4  head depth multiplier
    [16, 128],     # 5  head base channels          (int)
    [0.1, 0.9],    # 6  confidence threshold
    [0.1, 0.9],    # 7  NMS threshold
    [0.5, 2.0],    # 8  anchor scale P3
    [0.5, 2.0],    # 9  anchor scale P4
    [0.5, 2.0],    # 10 anchor scale P5
    [1, 4],        # 11 CSP blocks backbone         (int)
    [1, 3],        # 12 CSP blocks head             (int)
    [0, 1],        # 13 use separable conv          (binary)
    [0, 1],        # 14 use ghost modules           (binary)
], dtype=float)

DIM = len(BOUNDS)
LOW, HIGH = BOUNDS[:, 0].copy(), BOUNDS[:, 1].copy()
INT_DIMS = (0, 5, 11, 12)
BIN_DIMS = (13, 14)

def clip(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, LOW, HIGH)

def discretize(theta: np.ndarray) -> np.ndarray:
    """Canonical evaluated form: clip, round integer dims, snap binaries.
    This is the SAME function for every strategy (equal-footing requirement)."""
    t = clip(np.asarray(theta, dtype=float)).copy()
    for d in INT_DIMS:
        t[d] = round(t[d])
    for d in BIN_DIMS:
        t[d] = 1.0 if t[d] > 0.5 else 0.0
    return t

def sample_uniform(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    return rng.uniform(LOW, HIGH, size=(n, DIM))

def theta_to_json(theta: np.ndarray) -> str:
    return json.dumps([round(float(x), 6) for x in np.asarray(theta).tolist()])

def theta_from_json(s: str) -> np.ndarray:
    return np.array(json.loads(s), dtype=float)
