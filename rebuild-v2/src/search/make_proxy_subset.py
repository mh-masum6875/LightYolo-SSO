"""Create the fixed 25 percent stratified proxy-train subset (Master Plan 6.2).
Shared by ALL strategies: data/splits/proxy_train_25pct.txt, plus a proxy
data yaml pointing train at the subset and val at the full loc_hard val split.

Usage: python -m src.search.make_proxy_subset [--frac 0.25] [--seed 0]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = os.path.join(ROOT, "data", "benchmarks", "loc_hard")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(os.path.join(BENCH, "train.txt")) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    lbl_dir = os.path.join(BENCH, "labels")
    faulty, clean = [], []
    for i in ids:
        p = os.path.join(lbl_dir, i + ".txt")
        (faulty if os.path.getsize(p) > 0 else clean).append(i)
    rng = np.random.default_rng(a.seed + 2000)
    keep = []
    for group in (faulty, clean):
        g = np.array(group)
        rng.shuffle(g)
        keep += g[: int(round(a.frac * len(g)))].tolist()
    keep = sorted(keep)
    out = os.path.join(ROOT, "data", "splits", "proxy_train_25pct.txt")
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(keep) + "\n")
    paths = os.path.join(BENCH, "proxy_train_paths.txt")
    with open(paths, "w", newline="\n") as f:
        f.write("\n".join(f"./images/{i}.png" for i in keep) + "\n")
    n_f = sum(1 for i in keep if os.path.getsize(os.path.join(lbl_dir, i + ".txt")) > 0)
    print(f"proxy subset: {len(keep)} windows ({n_f} faulty, {len(keep)-n_f} clean) -> {out}")

if __name__ == "__main__":
    main()
