"""Window <-> image encoding and YOLO box generation for WSN-FaultBench.

Geometry (Plain-Language Guide 3.2): window = 64 time steps x 4 modalities,
image = 256x256 grayscale. Modality m owns pixel rows [64m, 64m+64);
time step t owns pixel columns [4t, 4t+4). Values are grayscale 0-255.
"""
import numpy as np

W_LEN = 64          # time steps per window
N_MOD = 4           # modalities
IMG = 256
ROWS_PER_MOD = IMG // N_MOD   # 64
COLS_PER_T = IMG // W_LEN     # 4
PAD = 2
MIN_BOX = 8
FAULT_CLASSES = {"spike": 0, "stuck": 1, "drift": 2, "noise": 3}

def window_to_image(win):
    """win: (64,4) float array on 0-255 scale -> (256,256) uint8 image."""
    assert win.shape == (W_LEN, N_MOD)
    img = np.empty((IMG, IMG), dtype=np.uint8)
    v = np.clip(np.round(win), 0, 255).astype(np.uint8)
    for m in range(N_MOD):
        band = np.repeat(v[:, m], COLS_PER_T)[None, :]          # (1,256) over time
        img[m*ROWS_PER_MOD:(m+1)*ROWS_PER_MOD, :] = np.repeat(band, ROWS_PER_MOD, axis=0)
    return img

def fault_to_box(modality, t_start, t_end):
    """Pixel box for a fault. t_end inclusive. Returns (x0, y0, x1, y1) exclusive right/bottom."""
    y0 = modality * ROWS_PER_MOD
    y1 = (modality + 1) * ROWS_PER_MOD
    x0 = t_start * COLS_PER_T
    x1 = (t_end + 1) * COLS_PER_T
    # pad and clip
    x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
    x1, y1 = min(IMG, x1 + PAD), min(IMG, y1 + PAD)
    # enforce minimum size
    if x1 - x0 < MIN_BOX:
        grow = MIN_BOX - (x1 - x0)
        x0 = max(0, x0 - grow // 2); x1 = min(IMG, x0 + MIN_BOX)
    if y1 - y0 < MIN_BOX:
        grow = MIN_BOX - (y1 - y0)
        y0 = max(0, y0 - grow // 2); y1 = min(IMG, y0 + MIN_BOX)
    return x0, y0, x1, y1

def box_to_yolo(cls_id, x0, y0, x1, y1):
    cx = (x0 + x1) / 2 / IMG
    cy = (y0 + y1) / 2 / IMG
    w = (x1 - x0) / IMG
    h = (y1 - y0) / IMG
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
