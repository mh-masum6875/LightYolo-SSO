"""WP3 data adapters. Three tasks (Master Plan 7):
  continuity - original binary task: IBRL windows, labels from the stability
               rule (std<=10%, range<=30%), WP0 frozen splits (24,853/3,105/3,109).
  lwsndr     - transfer check: humidity+temperature windows, real labels.
  loc_hard   - window-level binary view of WSN-FaultBench (faulty iff any fault),
               for image classifiers only ("what classification cannot localise").

Returns raw windows (N,64,4 float 0-255), labels (N,), node ids (N,) and split
indices. Heavy work is cached under data/benchmarks/wp3_cache/.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_IBRL = os.path.join(ROOT, "..", "Raw_Data", "data.txt")
CACHE_DIR = os.path.join(ROOT, "data", "benchmarks", "wp3_cache")
SPLITS_DIR = os.path.join(ROOT, "data", "splits")

def _continuity_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "continuity_windows.npz")

def load_continuity():
    """All IBRL windows in original order with rule labels; splits = WP0 frozen
    index files. Window construction follows the original conversion
    conventions (global min-max 0-255, per-node non-overlapping 64x4)."""
    from src.injection import ibrl
    from src.baselines.rule import rule_is_normal
    cache = _continuity_cache()
    if os.path.exists(cache):
        z = np.load(cache)
        wins, nodes, labels = z["wins"], z["nodes"], z["labels"]
    else:
        df = ibrl.load_raw(RAW_IBRL)
        params = ibrl.normalization_params(df)
        df = ibrl.normalize(df, params)
        wins, nodes = [], []
        for node, _i, win in ibrl.windows_per_node(df):
            wins.append(win); nodes.append(node)
        wins = np.stack(wins).astype(np.float32); nodes = np.array(nodes)
        labels = np.array([0 if rule_is_normal(w) else 1 for w in wins], dtype=np.int64)
        np.savez_compressed(cache, wins=wins, nodes=nodes, labels=labels)
    splits = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(SPLITS_DIR, f"{s}.txt")) as f:
            splits[s] = np.array([int(ln) for ln in f if ln.strip()], dtype=np.int64)
    return wins, labels, nodes, splits

def load_faultstream(tier="hard"):
    """Raw-stream view of WSN-FaultBench for the cross-family baselines (R2.4).

    Reconstructs, bit-identically, the SAME faulted 64x4 windows that WSN-YOLO
    sees as images: clean base windows (base_windows.npz) + the loc_<tier>
    manifest (which windows get which faults) re-injected via the per-fault RNG,
    with the loc_<tier> frozen splits. Label = 1 iff the window carries any
    fault. No IBRL data.txt required; every baseline is thus judged on exactly
    the benchmark WSN-YOLO is evaluated on.
    """
    from src.injection.build_benchmark import read_manifest, apply_faults, CACHE
    bench = os.path.join(ROOT, "data", "benchmarks", f"loc_{tier}")
    z = np.load(CACHE)
    wins = z["wins"].astype(np.float32)          # (N,64,4), 0-255 clean base
    nodes = z["nodes"]
    rows = read_manifest(os.path.join(bench, "manifest.csv"))
    by_w = {}
    for r in rows:
        by_w.setdefault(r["window_id"], []).append(r)
    seed = rows[0]["seed"] if rows else 0
    labels = np.zeros(len(wins), dtype=np.int64)
    for wi, frows in by_w.items():
        apply_faults(wins[wi], frows, seed)      # in-place, same RNG as images
        labels[wi] = 1
    splits = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(bench, f"{s}.txt")) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        splits[s] = np.array([int(i.lstrip("w")) for i in ids], dtype=np.int64)
    return wins, labels, nodes, splits

def load_lwsndr():
    """Windows + real labels + frozen block-level splits from the lwsndr benchmark."""
    import csv
    bench = os.path.join(ROOT, "data", "benchmarks", "lwsndr")
    from src.injection import lwsndr as L
    streams = L.load_streams(os.path.join(ROOT, "..", "Raw_Data"))
    params = L.normalization_params(streams)
    wins, labels, motes = [], [], []
    for mote, win, lab in L.windows(streams, params):
        wins.append(win); labels.append(lab); motes.append(mote)
    wins = np.stack(wins).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    motes = np.array(motes)
    splits = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(bench, f"{s}.txt")) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        splits[s] = np.array([int(i.lstrip("w")) for i in ids], dtype=np.int64)
    return wins, labels, motes, splits

def load_loc_hard_binary():
    """Window-level (image, faulty-or-not) view of loc_hard for image classifiers."""
    bench = os.path.join(ROOT, "data", "benchmarks", "loc_hard")
    lbl_dir = os.path.join(bench, "labels")
    splits, labels, names = {}, {}, {}
    for s in ("train", "val", "test"):
        with open(os.path.join(bench, f"{s}.txt")) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        splits[s] = ids
        for i in ids:
            labels[i] = 1 if os.path.getsize(os.path.join(lbl_dir, i + ".txt")) > 0 else 0
    return bench, splits, labels
