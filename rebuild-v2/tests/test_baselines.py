import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.metrics import (accuracy, f1, auroc, threshold_at_detection,
                                false_alarms_per_node_day, evaluate_scores)
from src.baselines.features import window_features, batch_features
from src.baselines.rule import rule_is_normal, rule_scores, STD_MAX, RANGE_MAX

def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert abs(auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12  # ties -> 0.5

def test_f1_and_accuracy():
    y, p = np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])
    assert accuracy(y, p) == 0.5
    assert abs(f1(y, p) - 0.5) < 1e-12

def test_threshold_meets_detection_on_val():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([rng.normal(0, 1, 500), rng.normal(2, 1, 500)])
    thr = threshold_at_detection(y, s, 0.95)
    assert ((s[y == 1] >= thr).mean()) >= 0.95

def test_false_alarm_units():
    from src.common.metrics import WINDOWS_PER_DAY
    n_nodes, n_windows, n_fp = 10, 434, 5
    y = np.zeros(n_windows, dtype=int)
    p = np.zeros(n_windows, dtype=int); p[:n_fp] = 1
    fa = false_alarms_per_node_day(y, p, n_nodes=n_nodes)
    days = n_windows / (WINDOWS_PER_DAY * n_nodes)
    assert abs(fa - n_fp / (n_nodes * days)) < 1e-9

def test_rule_perfect_on_continuity_by_construction():
    """Master Plan 7 acceptance: labels produced by the rule are recovered
    perfectly by the rule (the circularity the paper states openly)."""
    rng = np.random.default_rng(1)
    normal = 128 + rng.normal(0, 3, size=(50, 64, 4))            # tiny std/range
    anomalous = 128 + rng.normal(0, 40, size=(50, 64, 4))        # violates thresholds
    wins = np.clip(np.concatenate([normal, anomalous]), 0, 255)
    labels = np.array([0 if rule_is_normal(w) else 1 for w in wins])
    assert labels[:50].sum() == 0 and labels[50:].sum() == 50    # construction sane
    assert np.array_equal(rule_scores(wins), labels)             # perfect by construction
    res = evaluate_scores(labels, rule_scores(wins), labels, rule_scores(wins), n_nodes=5)
    assert res["accuracy"] == 1.0 and res["f1"] == 1.0

def test_features_shape_and_sensitivity():
    rng = np.random.default_rng(2)
    win = 128 + rng.normal(0, 2, size=(64, 4))
    f = window_features(win)
    assert f.shape == (12,)
    spiked = win.copy(); spiked[30, 1] += 50
    f2 = window_features(spiked)
    assert f2[1] > f[1] and f2[5] > f[5]        # std and range of modality 1 grew
    assert batch_features(np.stack([win, spiked])).shape == (2, 12)
