"""Build loc_lwsndr: a SECOND localized fault-detection benchmark from the LWSNDR
dataset (independent WSN source: humidity + temperature), using the identical
WSN-FaultBench methodology - spike/stuck/drift/noise injected at known
modality-time spans into clean LWSNDR windows, encoded as 256x256 images with
GENUINE bounding boxes. This validates the localization approach on data
independent of IBRL, without reintroducing the full-image-box weakness.

Usage:
  python -m src.injection.build_loc_lwsndr --raw <Raw_Data> --out data/benchmarks/loc_lwsndr
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.injection import lwsndr as L
from src.injection.fault_models import inject, draw_fault, fault_rng

N_MOD, W_LEN, IMG = 2, 64, 256
ROWS_PER_MOD, COLS_PER_T = IMG // N_MOD, IMG // W_LEN     # 128, 4
PAD, MIN_BOX = 2, 8
FAULT_CLASSES = {"spike": 0, "stuck": 1, "drift": 2, "noise": 3}
INJECT_FRAC, MAX_FAULTS = 0.45, 3


def fault_to_box(m, t0, t1):
    y0, y1 = m * ROWS_PER_MOD, (m + 1) * ROWS_PER_MOD
    x0, x1 = t0 * COLS_PER_T, (t1 + 1) * COLS_PER_T
    x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
    x1, y1 = min(IMG, x1 + PAD), min(IMG, y1 + PAD)
    if x1 - x0 < MIN_BOX:
        x1 = min(IMG, x0 + MIN_BOX)
    if y1 - y0 < MIN_BOX:
        y1 = min(IMG, y0 + MIN_BOX)
    return x0, y0, x1, y1


def box_to_yolo(cls, x0, y0, x1, y1):
    return (f"{cls} {((x0 + x1) / 2) / IMG:.6f} {((y0 + y1) / 2) / IMG:.6f} "
            f"{(x1 - x0) / IMG:.6f} {(y1 - y0) / IMG:.6f}")


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", choices=["hard", "easy"], default="easy",
                    help="fault intensity tier; easy = stronger, clearly-visible faults")
    ap.add_argument("--base", default="/home/user/mhmasum6875/yolov7_maidul/rebuild-v2/data/benchmarks/loc_lwsndr",
                    help="absolute DICC path to the out dir; used for the absolute-path lists/yaml")
    a = ap.parse_args()

    streams = L.load_streams(a.raw)
    params = L.normalization_params(streams)
    wins = [win.astype(np.float32) for _n, _i, win, lab in L.windows(streams, params) if lab == 0]
    wins = np.stack(wins)
    N = len(wins)

    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(N, int(round(INJECT_FRAC * N)), replace=False).tolist())
    rows = []
    for wi in chosen:
        used = []
        for _ in range(int(rng.integers(1, MAX_FAULTS + 1))):
            free = [m for m in range(N_MOD) if m not in used]
            if not free:
                break
            m = int(free[rng.integers(0, len(free))]); used.append(m)
            ft, t0, t1, k = draw_fault(rng, a.tier)
            rows.append(dict(window_id=wi, modality=m, fault_type=ft, t_start=t0,
                             t_end=t1, intensity_param=round(k, 6), seed=a.seed))

    by_w = {}
    for r in rows:
        by_w.setdefault(r["window_id"], []).append(r)

    img_dir = os.path.join(a.out, "images"); lbl_dir = os.path.join(a.out, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    for wi in range(N):
        win = wins[wi].copy()
        frows = by_w.get(wi, [])
        for r in frows:
            rr = fault_rng(a.seed, wi, r["modality"], r["t_start"], r["fault_type"])
            inject(win, r["fault_type"], r["modality"], r["t_start"], r["t_end"], r["intensity_param"], rr)
        Image.fromarray(L.window_to_image(win)).save(os.path.join(img_dir, f"w{wi:06d}.png"))
        lines = [box_to_yolo(FAULT_CLASSES[r["fault_type"]], *fault_to_box(r["modality"], r["t_start"], r["t_end"]))
                 for r in frows]
        with open(os.path.join(lbl_dir, f"w{wi:06d}.txt"), "w", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="\n") as f:
        cols = ["window_id", "modality", "fault_type", "t_start", "t_end", "intensity_param", "seed"]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow(r)

    faulty = set(by_w); clean = [i for i in range(N) if i not in faulty]
    rng2 = np.random.default_rng(a.seed + 1000)
    splits = {"train": [], "val": [], "test": []}
    for grp in (sorted(faulty), clean):
        idx = np.array(grp); rng2.shuffle(idx)
        n = len(idx); ntr, nva = int(0.8 * n), int(0.1 * n)
        splits["train"] += idx[:ntr].tolist()
        splits["val"] += idx[ntr:ntr + nva].tolist()
        splits["test"] += idx[ntr + nva:].tolist()
    for s, ids in splits.items():
        with open(os.path.join(a.out, f"{s}.txt"), "w", newline="\n") as f:
            f.write("\n".join(f"w{i:06d}" for i in sorted(ids)) + "\n")
        with open(os.path.join(a.out, f"{s}_paths.txt"), "w", newline="\n") as f:
            f.write("\n".join(f"./images/w{i:06d}.png" for i in sorted(ids)) + "\n")

    with open(os.path.join(a.out, "dataset.yaml"), "w", newline="\n") as f:
        f.write("path: .\ntrain: train_paths.txt\nval: val_paths.txt\ntest: test_paths.txt\n"
                "nc: 4\nnames: ['spike', 'stuck', 'drift', 'noise']\n")

    # absolute-path lists + yaml for yolov7/yolov5/ultralytics (which ignore path:)
    for s in ("train", "val", "test"):
        ids = [f"w{i:06d}" for i in sorted(splits[s])]
        with open(os.path.join(a.out, f"{s}_abs.txt"), "w", newline="\n") as f:
            f.write("\n".join(f"{a.base}/images/{i}.png" for i in ids) + "\n")
    with open(os.path.join(a.out, "dataset_abs.yaml"), "w", newline="\n") as f:
        f.write(f"train: {a.base}/train_abs.txt\nval: {a.base}/val_abs.txt\ntest: {a.base}/test_abs.txt\n"
                "nc: 4\nnames: ['spike', 'stuck', 'drift', 'noise']\n")

    print(f"loc_lwsndr: {N} windows, {len(faulty)} faulty ({100*len(faulty)/N:.0f}%), {len(rows)} faults")
    print("splits:", {s: len(v) for s, v in splits.items()})


if __name__ == "__main__":
    main()
