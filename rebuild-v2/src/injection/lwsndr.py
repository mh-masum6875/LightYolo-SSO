"""LWSNDR ingestion (Suthaharan et al. 2010) for transfer evaluation.

8 streams: {single,multi}-hop x {indoor,outdoor} x mote. Columns:
reading#, mote_id, humidity, temperature, label (0 normal, 1 introduced event).
Windows: 64 samples x 2 modalities; window label = 1 if ANY sample is labelled 1.
Encoding: 256x256 grayscale; humidity owns rows [0,128), temperature rows [128,256);
time step t owns cols [4t, 4t+4). Binary task in the continuity convention:
full-frame YOLO box, class 0 = normal, 1 = anomaly.
Normalization: per modality, 1-99 pctl -> 0-255, computed over all 8 streams.
"""
import os
import numpy as np

W_LEN, IMG = 64, 256
ROWS_PER_MOD, COLS_PER_T = 128, 4

def parse_file(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append((float(parts[2]), float(parts[3]), int(float(parts[4]))))
            except ValueError:
                continue
    return np.array(rows)  # (n, 3): humidity, temperature, label

def load_streams(raw_root):
    streams = {}
    specs = [("singlehop", os.path.join(raw_root, "SingleHopLabelledReadings", "SingleHopLabelledReadings")),
             ("multihop", os.path.join(raw_root, "MultiHopLebelledReadings", "MultiHopLebelledReadings"))]
    for tag, d in specs:
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt"):
                streams[f"{tag}_{fn.replace('-data.txt','')}"] = parse_file(os.path.join(d, fn))
    return streams

def normalization_params(streams):
    allv = np.concatenate([s[:, :2] for s in streams.values()])
    return {m: tuple(np.percentile(allv[:, m], [1, 99])) for m in range(2)}

def windows(streams, params):
    """Yield (stream_name, window_index, win[64,2] 0-255 float, label)."""
    for name, arr in sorted(streams.items()):
        vals = arr[:, :2].copy()
        for m in range(2):
            lo, hi = params[m]
            vals[:, m] = np.clip((vals[:, m] - lo) / (hi - lo), 0, 1) * 255.0
        labels = arr[:, 2]
        for i in range(len(arr) // W_LEN):
            seg = slice(i * W_LEN, (i + 1) * W_LEN)
            yield name, i, vals[seg], int(labels[seg].max())

def window_to_image(win):
    img = np.empty((IMG, IMG), dtype=np.uint8)
    v = np.clip(np.round(win), 0, 255).astype(np.uint8)
    for m in range(2):
        band = np.repeat(v[:, m], COLS_PER_T)[None, :]
        img[m*ROWS_PER_MOD:(m+1)*ROWS_PER_MOD, :] = np.repeat(band, ROWS_PER_MOD, axis=0)
    return img
