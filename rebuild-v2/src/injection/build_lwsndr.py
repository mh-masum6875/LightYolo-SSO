"""Build the LWSNDR transfer benchmark: images, binary YOLO labels, manifest, frozen splits.

Windows are OVERLAPPING (stride 16) because introduced events cover only 0.8% of
samples; non-overlapping windows yield just 10 anomalous windows. To prevent
train/test leakage from overlap, splitting is done at the level of contiguous
128-sample BLOCKS per stream: windows never cross block boundaries and every
window in a block goes to the same split. Blocks containing any anomalous sample
are stratified separately from clean blocks (seeded, frozen).
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.injection import lwsndr
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "..", "Raw_Data")
OUT = os.path.join(ROOT, "data", "benchmarks", "lwsndr")
SEED, W_LEN, STRIDE, BLOCK = 0, 64, 16, 128

def main():
    streams = lwsndr.load_streams(RAW)
    params = lwsndr.normalization_params(streams)
    img_dir, lbl_dir = os.path.join(OUT, "images"), os.path.join(OUT, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    manifest, block_names, block_has_anom = [], {}, {}
    for sname, arr in sorted(streams.items()):
        vals = arr[:, :2].copy()
        for m in range(2):
            lo, hi = params[m]
            vals[:, m] = np.clip((vals[:, m] - lo) / (hi - lo), 0, 1) * 255.0
        labels = arr[:, 2]
        for b in range(len(arr) // BLOCK):
            bkey = (sname, b)
            block_names[bkey] = []
            block_has_anom[bkey] = False
            b0 = b * BLOCK
            for t0 in range(b0, b0 + BLOCK - W_LEN + 1, STRIDE):
                win, wl = vals[t0:t0 + W_LEN], int(labels[t0:t0 + W_LEN].max())
                name = f"{sname}_t{t0:06d}"
                Image.fromarray(lwsndr.window_to_image(win)).save(os.path.join(img_dir, name + ".png"))
                with open(os.path.join(lbl_dir, name + ".txt"), "w", newline="\n") as f:
                    f.write(f"{wl} 0.5 0.5 1.0 1.0\n")
                manifest.append(dict(name=name, stream=sname, t_start=t0, block=b, label=wl))
                block_names[bkey].append(name)
                if wl: block_has_anom[bkey] = True
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["name", "stream", "t_start", "block", "label"], lineterminator="\n")
        w.writeheader(); w.writerows(manifest)
    rng = np.random.default_rng(SEED + 2000)
    anom_blocks = sorted([k for k, v in block_has_anom.items() if v])
    clean_blocks = sorted([k for k, v in block_has_anom.items() if not v])
    splits = {"train": [], "val": [], "test": []}
    for group in (anom_blocks, clean_blocks):
        idx = list(group); rng.shuffle(idx)
        n = len(idx)
        # guarantee at least one block of each group in val and test
        n_va = max(1, round(0.1 * n))
        n_te = max(1, round(0.1 * n))
        n_tr = n - n_va - n_te
        for k in idx[:n_tr]: splits["train"] += block_names[k]
        for k in idx[n_tr:n_tr + n_va]: splits["val"] += block_names[k]
        for k in idx[n_tr + n_va:]: splits["test"] += block_names[k]
    lab = {m["name"]: m["label"] for m in manifest}
    for s, ns in splits.items():
        with open(os.path.join(OUT, f"{s}.txt"), "w", newline="\n") as f:
            f.write("\n".join(sorted(ns)) + "\n")
        with open(os.path.join(OUT, f"{s}_paths.txt"), "w", newline="\n") as f:
            f.write("\n".join(f"./images/{n}.png" for n in sorted(ns)) + "\n")
    with open(os.path.join(OUT, "dataset.yaml"), "w", newline="\n") as f:
        f.write("path: .\ntrain: train_paths.txt\nval: val_paths.txt\ntest: test_paths.txt\nnc: 2\nnames: ['normal', 'anomaly']\n")
    stats = {s: (len(ns), sum(lab[n] for n in ns)) for s, ns in splits.items()}
    print(f"windows: {len(manifest)} | anomalous: {sum(m['label'] for m in manifest)} | "
          f"anom blocks: {len(anom_blocks)}/{len(block_names)} | split (n, anom): {stats}")

if __name__ == "__main__":
    main()
