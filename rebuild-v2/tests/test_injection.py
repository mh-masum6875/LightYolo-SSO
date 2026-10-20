import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.injection.encode import (window_to_image, fault_to_box, box_to_yolo,
                                  ROWS_PER_MOD, COLS_PER_T, IMG, PAD)
from src.injection.fault_models import inject, fault_rng
from src.injection.build_benchmark import plan_faults, apply_faults, write_manifest, read_manifest
import tempfile

def test_box_decodes_to_exact_injected_pixels():
    """Plan 5.2: a synthetic window with a known span decodes to exactly the injected pixels."""
    win = np.full((64, 4), 128.0)
    modality, t0, t1 = 1, 20, 21           # guide worked example: humidity, steps 20-21
    rng = fault_rng(0, 0, modality, t0, "spike")
    inject(win, "spike", modality, t0, t1, 8.0, rng)
    img = window_to_image(win)
    changed = np.argwhere(img != 128)
    rows, cols = changed[:, 0], changed[:, 1]
    assert rows.min() == modality * ROWS_PER_MOD == 64
    assert rows.max() == (modality + 1) * ROWS_PER_MOD - 1 == 127
    assert cols.min() == t0 * COLS_PER_T == 80
    assert cols.max() == (t1 + 1) * COLS_PER_T - 1 == 87
    x0, y0, x1, y1 = fault_to_box(modality, t0, t1)
    assert (x0, y0, x1, y1) == (80 - PAD, 64 - PAD, 88 + PAD, 128 + PAD)

def test_min_box_size():
    x0, y0, x1, y1 = fault_to_box(0, 0, 0)   # 1-step spike at t=0
    assert x1 - x0 >= 8 and y1 - y0 >= 8 and x0 >= 0 and y1 <= IMG

def test_yolo_line_format():
    line = box_to_yolo(2, 78, 62, 90, 130)
    cls, cx, cy, w, h = line.split()
    assert cls == "2" and 0 < float(cx) < 1 and 0 < float(w) < 1

def test_manifest_regeneration_bit_identical():
    """Plan 5.2: regeneration from the manifest must be bit-identical."""
    rng = np.random.default_rng(42)
    wins = rng.uniform(100, 140, size=(50, 64, 4))
    nodes = np.arange(50) % 5
    rows = plan_faults(wins, nodes, "hard", seed=7)
    assert len(rows) > 0
    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, "manifest.csv")
        write_manifest(rows, mpath)
        rows2 = read_manifest(mpath)
    by_w, by_w2 = {}, {}
    for r in rows: by_w.setdefault(r["window_id"], []).append(r)
    for r in rows2: by_w2.setdefault(r["window_id"], []).append(r)
    assert sorted(by_w) == sorted(by_w2)
    for wi in by_w:
        a = apply_faults(wins[wi].copy(), by_w[wi], seed=7)
        b = apply_faults(wins[wi].copy(), by_w2[wi], seed=7)
        assert np.array_equal(a, b), f"window {wi} not bit-identical"
        assert np.array_equal(window_to_image(a), window_to_image(b))

def test_stuck_is_constant_and_deviated():
    """Revision 2: stuck fault must hold ONE constant value that deviates from
    the clean signal by k*sigma, so it is detectable on screened-stable windows."""
    rng0 = np.random.default_rng(0)
    win = 128.0 + rng0.normal(0, 2.0, size=(64, 4))
    clean = win.copy()
    m, t0, t1, k = 2, 10, 20, 2.0
    rng = fault_rng(0, 0, m, t0, "stuck")
    inject(win, "stuck", m, t0, t1, k, rng)
    seg = win[t0:t1 + 1, m]
    assert np.all(seg == seg[0]), "stuck segment must be constant"
    sig = max(float(np.std(clean[:, m])), 2.0)
    assert abs(seg[0] - clean[t0, m]) >= 0.99 * k * sig, "stuck value must deviate"
    assert np.array_equal(win[:t0, m], clean[:t0, m])
    assert np.array_equal(win[:, :m], clean[:, :m])

def test_no_modality_overlap():
    rng = np.random.default_rng(1)
    wins = rng.uniform(100, 140, size=(200, 64, 4))
    nodes = np.zeros(200, dtype=int)
    rows = plan_faults(wins, nodes, "easy", seed=3)
    seen = {}
    for r in rows:
        key = (r["window_id"], r["modality"])
        assert key not in seen, "two faults on same modality in same window"
        seen[key] = True

def test_determinism_same_seed_same_plan():
    rng = np.random.default_rng(9)
    wins = rng.uniform(100, 140, size=(80, 64, 4))
    nodes = np.zeros(80, dtype=int)
    assert plan_faults(wins, nodes, "hard", 5) == plan_faults(wins, nodes, "hard", 5)
