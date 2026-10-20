"""Place WSN-YOLO in the cross-family table (Table 5) by scoring it on the SAME
binary window-level task as the anomaly-detection baselines. WSN-YOLO outputs
detections; we reduce each window to a single anomaly score = the maximum
detection confidence in that window (0 if nothing is detected). Ground truth is
window-level faulty/clean (a window is faulty iff its loc_hard label is
non-empty). The decision threshold is swept on val and evaluated once on test,
exactly like the WP3 baselines, giving accuracy / F1 / AUROC / false-alarms.

Run on DICC from the yolov7 repo root (yolov7_maidul env):
  python rebuild-v2/src/baselines/wsnyolo_auroc.py \
     --weights runs/frontier/arch_07/weights/best.pt \
     --data   rebuild-v2/data/benchmarks/loc_hard/dataset_yolov5.yaml \
     --loc-hard rebuild-v2/data/benchmarks/loc_hard
"""
from __future__ import annotations
import argparse, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # rebuild-v2 on path


def run_test(ydir, weights, data, task, img, name):
    cmd = [sys.executable, "test.py", "--weights", weights, "--data", data,
           "--task", task, "--img-size", str(img), "--conf-thres", "0.001",
           "--iou-thres", "0.6", "--save-txt", "--save-conf",
           "--project", "runs", "--name", name, "--exist-ok"]
    print("[test]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ydir, check=True)
    return os.path.join(ydir, "runs", name, "labels")


def ids_of(loc_hard, split):
    return [l.strip() for l in open(os.path.join(loc_hard, f"{split}.txt")) if l.strip()]


def gt_and_scores(loc_hard, pred_dir, split):
    ids = ids_of(loc_hard, split)
    y, s = [], []
    for i in ids:
        lab = os.path.join(loc_hard, "labels", i + ".txt")
        y.append(1 if (os.path.exists(lab) and os.path.getsize(lab) > 0) else 0)
        pred = os.path.join(pred_dir, i + ".txt")
        m = 0.0
        if os.path.exists(pred):
            for line in open(pred):
                p = line.split()
                if len(p) >= 6:
                    m = max(m, float(p[5]))       # class cx cy w h conf
        s.append(m)
    return np.array(y, dtype=np.int64), np.array(s, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolov7-dir", default=".")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--loc-hard", required=True)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--n-nodes", type=int, default=55)
    ap.add_argument("--db")
    a = ap.parse_args()
    ydir = os.path.abspath(a.yolov7_dir)

    vp = run_test(ydir, a.weights, a.data, "val", a.img, "wsnyolo_auroc_val")
    tp = run_test(ydir, a.weights, a.data, "test", a.img, "wsnyolo_auroc_test")
    yv, sv = gt_and_scores(a.loc_hard, vp, "val")
    yt, st = gt_and_scores(a.loc_hard, tp, "test")

    from src.common.metrics import evaluate_scores
    res = evaluate_scores(yv, sv, yt, st, n_nodes=a.n_nodes)
    print(f"\ntest windows: {len(yt)}, faulty {int(yt.sum())} ({100*yt.mean():.0f}%)")
    print("WSN-YOLO window-level:", {k: round(float(v), 4) for k, v in res.items()})

    if a.db:
        from src.common.db import log_metric
        rid = "wp3_wsnyolo_loc_hard_bin"
        for k in ("accuracy", "f1", "auroc", "false_alarms_per_node_day", "recall_test"):
            log_metric(a.db, rid, k, float(res[k]), wp="WP3", task="loc_hard",
                       dataset="loc_hard", method="wsn_yolo", model="wsn_yolo",
                       seed=0, split="test")
        print("logged ->", a.db)


if __name__ == "__main__":
    main()
