"""WP3 driver: one (model, task, seed) evaluation, logged to the database.

Usage (DICC):
  python -m src.baselines.run_baseline --model lstm_ae --task continuity --seed 0
  python -m src.baselines.run_baseline --model rule --task continuity --seed 0
Tasks: continuity, lwsndr (all raw/feature baselines); image classifiers
(mobilenetv3s, shufflenet) additionally on loc_hard via run_image_clf.py;
yolov8n via run_yolov8.py. 5 seeds for learned models (Master Plan 7).
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.common.metrics import evaluate_scores
from src.baselines import data as D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_MODELS = ("lstm_ae", "dense_ae", "cnn1d", "gru_clf", "ocsvm", "iforest", "rule")

def get_scores(model_name, wins, labels, nodes, splits, seed):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    if model_name == "rule":
        from src.baselines.rule import rule_scores
        return rule_scores(wins[va]), rule_scores(wins[te]), 0.0
    from src.baselines.models import MODELS
    m = MODELS[model_name]()
    t0 = time.time()
    if model_name == "ocsvm":
        m.fit(wins[tr], labels[tr], seed=seed, val=(wins[va], labels[va]))
    else:
        m.fit(wins[tr], labels[tr], seed=seed)
    return m.score(wins[va]), m.score(wins[te]), (time.time() - t0) / 3600.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=RAW_MODELS, required=True)
    ap.add_argument("--task", choices=("faultstream", "continuity", "lwsndr"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default=os.path.join(ROOT, "results", "results.sqlite"))
    a = ap.parse_args()

    if a.task == "faultstream":
        wins, labels, nodes, splits = D.load_faultstream("hard")
    elif a.task == "continuity":
        wins, labels, nodes, splits = D.load_continuity()
    else:
        wins, labels, nodes, splits = D.load_lwsndr()
    n_nodes = len(set(np.asarray(nodes).tolist()))
    s_val, s_test, gpu_h = get_scores(a.model, wins, labels, nodes, splits, a.seed)
    res = evaluate_scores(labels[splits["val"]], s_val,
                          labels[splits["test"]], s_test, n_nodes)
    run_id = f"wp3_{a.model}_{a.task}_s{a.seed}"
    from src.common.db import log_metric
    for k in ("accuracy", "f1", "auroc", "false_alarms_per_node_day", "recall_test", "threshold"):
        log_metric(a.db, run_id, k, float(res[k]), wp="WP3", task=a.task,
                   dataset=a.task, method=a.model, model=a.model, seed=a.seed,
                   split="test", gpu_hours=gpu_h if k == "accuracy" else None)
    print(run_id, {k: round(float(v), 4) for k, v in res.items()})

if __name__ == "__main__":
    main()
