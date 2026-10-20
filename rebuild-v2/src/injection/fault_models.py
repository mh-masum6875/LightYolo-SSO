"""Fault models for WSN-FaultBench: spike, stuck-at, drift, noise.

Taxonomy: Sharma, Golubchik, Govindan, ACM TOSN 2010.
Local sigma = pre-injection std of the affected modality within the window
(floored at SIGMA_FLOOR grayscale units so faults stay visible on ultra-clean windows).
Every fault is bit-identically reproducible from its manifest row: the per-fault
RNG is seeded from (global_seed, window_id, modality, t_start, fault_code).
"""
import numpy as np

SIGMA_FLOOR = 2.0  # grayscale units (0-255 scale)
FAULT_CODES = {"spike": 0, "stuck": 1, "drift": 2, "noise": 3}

# tier -> fault_type -> dict(k=intensity range, dur=duration range in samples)
# REVISION 2 (2026-07-17, gate attempt 1 = 0.6887 TOO HARD):
#   Confusion matrix showed the miss was class-specific: stuck recall 0.05
#   (flat segment on screened-stable windows is invisible by construction),
#   spike 0.69, drift 0.74, noise 0.96.
#   - stuck: redesigned as stuck-at-deviated-constant (Sharma et al. 2010
#     "constant" fault): value = x[t_start] + sign*k*sigma. k added below.
#   - spike: widened one step (2,4)->(3,5), matching Master Plan 5.1 spec.
#   - drift: widened one step (1.5,3)->(2,4).
#   - noise: unchanged.
TIERS = {
    "hard": {
        "spike": {"k": (3.0, 5.0),  "dur": (1, 3)},
        "stuck": {"k": (1.5, 3.0),  "dur": (8, 16)},
        "drift": {"k": (2.0, 4.0),  "dur": (16, 32)},
        "noise": {"k": (1.5, 3.0),  "dur": (8, 24)},
    },
    "easy": {
        "spike": {"k": (6.0, 12.0), "dur": (2, 6)},
        "stuck": {"k": (4.0, 8.0),  "dur": (24, 48)},
        "drift": {"k": (5.0, 10.0), "dur": (32, 64)},
        "noise": {"k": (4.0, 8.0),  "dur": (16, 48)},
    },
}

def fault_rng(global_seed, window_id, modality, t_start, fault_type):
    return np.random.default_rng(
        [global_seed, window_id, modality, t_start, FAULT_CODES[fault_type]])

def local_sigma(win, modality):
    return max(float(np.std(win[:, modality])), SIGMA_FLOOR)

def inject(win, fault_type, modality, t_start, t_end, k, rng):
    """Apply fault in-place on win (64,4 float, 0-255 scale). t_end inclusive."""
    sig = local_sigma(win, modality)
    seg = slice(t_start, t_end + 1)
    n = t_end - t_start + 1
    if fault_type == "spike":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        win[seg, modality] += sign * k * sig
    elif fault_type == "stuck":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        win[seg, modality] = win[t_start, modality] + sign * k * sig
    elif fault_type == "drift":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        ramp = np.linspace(0.0, sign * k * sig, n)
        win[seg, modality] += ramp
    elif fault_type == "noise":
        win[seg, modality] += rng.normal(0.0, k * sig, n)
    else:
        raise ValueError(fault_type)
    np.clip(win[:, modality], 0, 255, out=win[:, modality])
    return win

def draw_fault(rng, tier, t_max=64):
    """Draw (fault_type, t_start, t_end, k) uniformly from tier ranges."""
    ftype = ["spike", "stuck", "drift", "noise"][rng.integers(0, 4)]
    spec = TIERS[tier][ftype]
    dur = int(rng.integers(spec["dur"][0], spec["dur"][1] + 1))
    dur = min(dur, t_max)
    t_start = int(rng.integers(0, t_max - dur + 1))
    k = float(rng.uniform(spec["k"][0], spec["k"][1]))
    return ftype, t_start, t_start + dur - 1, k
