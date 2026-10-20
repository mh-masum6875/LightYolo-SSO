"""IBRL loading, cleaning, normalization, windowing for WSN-FaultBench.

Raw format (data.txt): date time epoch moteid temperature humidity light voltage.
Cleaning: drop physically implausible readings (documented in benchmark card).
Normalization: per modality, 1st-99th percentile clip then min-max to 0-255
(robust to the known IBRL outliers; computed once over the cleaned dataset).
Windows: per node, time-ordered, non-overlapping, 64 samples x 4 modalities.
Clean-base screening (plan 5.2): std <= 5 AND range <= 15 grayscale units per
modality (half the original labelling thresholds of 10/30 from IBRLConversion.py).
"""
import numpy as np
import pandas as pd

MODALITIES = ["temperature", "humidity", "light", "voltage"]
PLAUSIBLE = {"temperature": (0, 50), "humidity": (0, 100),
             "light": (0, 2000), "voltage": (1.5, 3.5)}
STD_MAX, RANGE_MAX = 5.0, 15.0   # clean-base screening, grayscale units
W_LEN = 64

def load_raw(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c",
                     names=["date", "time", "epoch", "moteid",
                            "temperature", "humidity", "light", "voltage"],
                     on_bad_lines="skip")
    df = df.dropna(subset=["moteid"] + MODALITIES)
    df["moteid"] = df["moteid"].astype(int)
    for m, (lo, hi) in PLAUSIBLE.items():
        df = df[(df[m] >= lo) & (df[m] <= hi)]
    return df

def normalization_params(df):
    params = {}
    for m in MODALITIES:
        lo, hi = np.percentile(df[m].to_numpy(), [1, 99])
        params[m] = (float(lo), float(hi))
    return params

def normalize(df, params):
    out = df.copy()
    for m in MODALITIES:
        lo, hi = params[m]
        out[m] = np.clip((df[m] - lo) / (hi - lo), 0, 1) * 255.0
    return out

def windows_per_node(df):
    """Yield (node_id, window_index_within_node, win[64,4] float array)."""
    for node, g in df.sort_values(["moteid", "epoch"]).groupby("moteid"):
        arr = g[MODALITIES].to_numpy(dtype=np.float64)
        n_win = len(arr) // W_LEN
        for i in range(n_win):
            yield int(node), i, arr[i*W_LEN:(i+1)*W_LEN].copy()

def is_clean(win):
    for c in range(win.shape[1]):
        col = win[:, c]
        if np.std(col) > STD_MAX or (col.max() - col.min()) > RANGE_MAX:
            return False
    return True
