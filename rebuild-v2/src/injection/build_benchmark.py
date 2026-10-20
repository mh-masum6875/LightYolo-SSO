"""Build WSN-FaultBench loc_hard / loc_easy from IBRL clean-base windows.

Usage:
  python -m src.injection.build_benchmark --tier hard --seed 0            # full build
  python -m src.injection.build_benchmark --tier hard --from-manifest M   # regenerate
  ... --cache-only        # only compute+cache clean base windows
  ... --chunk i --nchunks n   # write images for chunk i of n (resumable)

Spec (Master Plan v2 5.2): 45% of base windows receive 1-3 faults; no two
faults overlap on the same modality. All randomness from one numpy Generator
seeded per build; per-fault reproducibility via fault_rng(). Manifest row:
window_id, node_id, modality, fault_type, t_start, t_end, intensity_param, tier, seed.
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.injection import ibrl
from src.injection.fault_models import inject, draw_fault, fault_rng, TIERS
from src.injection.encode import window_to_image, fault_to_box, box_to_yolo, FAULT_CLASSES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "..", "Raw_Data", "data.txt")
CACHE = os.path.join(ROOT, "data", "benchmarks", "base_windows.npz")
INJECT_FRAC = 0.45
MAX_FAULTS = 3

def load_base_windows():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["wins"], z["nodes"], z["widx"]
    df = ibrl.load_raw(RAW)
    params = ibrl.normalization_params(df)
    df = ibrl.normalize(df, params)
    wins, nodes, widx = [], [], []
    for node, i, win in ibrl.windows_per_node(df):
        if ibrl.is_clean(win):
            wins.append(win); nodes.append(node); widx.append(i)
    wins = np.stack(wins); nodes = np.array(nodes); widx = np.array(widx)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, wins=wins, nodes=nodes, widx=widx,
        norm=np.array([params[m] for m in ibrl.MODALITIES]))
    return wins, nodes, widx

def plan_faults(wins, nodes, tier, seed):
    """Decide which windows get which faults. Returns manifest rows."""
    rng = np.random.default_rng(seed)
    n = len(wins)
    n_inject = int(round(INJECT_FRAC * n))
    chosen = rng.choice(n, size=n_inject, replace=False)
    rows = []
    for wi in sorted(chosen.tolist()):
        n_faults = int(rng.integers(1, MAX_FAULTS + 1))
        used_mods = []
        for _ in range(n_faults):
            free = [m for m in range(4) if m not in used_mods]
            if not free:
                break
            modality = int(free[rng.integers(0, len(free))])
            used_mods.append(modality)
            ftype, t0, t1, k = draw_fault(rng, tier)
            rows.append(dict(window_id=int(wi), node_id=int(nodes[wi]),
                             modality=modality, fault_type=ftype, t_start=t0,
                             t_end=t1, intensity_param=round(k, 6), tier=tier, seed=seed))
    return rows

def read_manifest(path):
    with open(path) as f:
        return [dict(window_id=int(r["window_id"]), node_id=int(r["node_id"]),
                     modality=int(r["modality"]), fault_type=r["fault_type"],
                     t_start=int(r["t_start"]), t_end=int(r["t_end"]),
                     intensity_param=float(r["intensity_param"]), tier=r["tier"],
                     seed=int(r["seed"])) for r in csv.DictReader(f)]

def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["window_id", "node_id", "modality", "fault_type", "t_start",
            "t_end", "intensity_param", "tier", "seed"]
    with open(path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

def apply_faults(win, rows, seed):
    for r in rows:
        rng = fault_rng(seed, r["window_id"], r["modality"], r["t_start"], r["fault_type"])
        inject(win, r["fault_type"], r["modality"], r["t_start"], r["t_end"],
               r["intensity_param"], rng)
    return win

def build_images(wins, rows, seed, out_dir, chunk=0, nchunks=1):
    from PIL import Image
    by_window = {}
    for r in rows:
        by_window.setdefault(r["window_id"], []).append(r)
    img_dir = os.path.join(out_dir, "images"); lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    ids = list(range(len(wins)))[chunk::nchunks]
    done = 0
    for wi in ids:
        name = f"w{wi:06d}"
        ipath = os.path.join(img_dir, name + ".png")
        if os.path.exists(ipath):
            continue
        win = wins[wi].copy()
        frows = by_window.get(wi, [])
        if frows:
            apply_faults(win, frows, seed)
        Image.fromarray(window_to_image(win)).save(ipath)
        lines = []
        for r in frows:
            box = fault_to_box(r["modality"], r["t_start"], r["t_end"])
            lines.append(box_to_yolo(FAULT_CLASSES[r["fault_type"]], *box))
        with open(os.path.join(lbl_dir, name + ".txt"), "w", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        done += 1
    return done

def make_benchmark_splits(n_windows, rows, seed, out_dir):
    """Frozen 80/10/10 split stratified by faulty/clean, seeded, committed."""
    rng = np.random.default_rng(seed + 1000)
    faulty = sorted({r["window_id"] for r in rows})
    clean = [i for i in range(n_windows) if i not in set(faulty)]
    splits = {"train": [], "val": [], "test": []}
    for group in (faulty, clean):
        idx = np.array(group); rng.shuffle(idx)
        n = len(idx); n_tr, n_va = int(0.8 * n), int(0.1 * n)
        splits["train"] += idx[:n_tr].tolist()
        splits["val"] += idx[n_tr:n_tr + n_va].tolist()
        splits["test"] += idx[n_tr + n_va:].tolist()
    for s, ids in splits.items():
        with open(os.path.join(out_dir, f"{s}.txt"), "w", newline="\n") as f:
            f.write("\n".join(f"w{i:06d}" for i in sorted(ids)) + "\n")
    return {s: len(v) for s, v in splits.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["hard", "easy"], default="hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-manifest")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    a = ap.parse_args()

    wins, nodes, widx = load_base_windows()
    print(f"base windows: {len(wins)} from {len(set(nodes.tolist()))} nodes")
    if a.cache_only:
        return
    out_dir = os.path.join(ROOT, "data", "benchmarks", f"loc_{a.tier}")
    mpath = os.path.join(out_dir, "manifest.csv")
    if a.from_manifest:
        rows = read_manifest(a.from_manifest)
    else:
        rows = plan_faults(wins, nodes, a.tier, a.seed)
        write_manifest(rows, mpath)
        print(f"manifest: {len(rows)} faults over {len({r['window_id'] for r in rows})} windows -> {mpath}")
    if a.plan_only:
        return
    done = build_images(wins, rows, a.seed, out_dir, a.chunk, a.nchunks)
    print(f"images written this run: {done}")
    if a.chunk == a.nchunks - 1:
        counts = make_benchmark_splits(len(wins), rows, a.seed, out_dir)
        print("splits:", counts)

if __name__ == "__main__":
    main()
