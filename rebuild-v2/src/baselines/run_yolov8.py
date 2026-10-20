"""WP3 R4 baseline: off-the-shelf yolov8n (ultralytics, default hyperparameters,
same data and splits) on loc_hard (detection) and continuity (classification
via detect-then-threshold is NOT used; continuity uses the curated detection
dataset exactly as the legacy runs did).

Usage (DICC, pip install ultralytics):
  python -m src.baselines.run_yolov8 --task loc_hard --seed 0
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("loc_hard", "continuity"), default="loc_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data")
    ap.add_argument("--db", default=os.path.join(ROOT, "results", "results.sqlite"))
    a = ap.parse_args()
    from ultralytics import YOLO
    # ultralytics resolves dataset paths differently from yolov7; use the
    # absolute-path config (dataset_yolov8.yaml) for loc_hard.
    default_yaml = ("dataset_yolov8.yaml" if a.task == "loc_hard" else "dataset.yaml")
    data = a.data or os.path.join(ROOT, "data", "benchmarks",
                                  "loc_hard" if a.task == "loc_hard" else "continuity",
                                  default_yaml)
    t0 = time.time()
    model = YOLO("yolov8n.pt")
    r = model.train(data=data, epochs=a.epochs, imgsz=640, seed=a.seed,
                    project=os.path.join(ROOT, "results", "wp3_yolov8"),
                    name=f"{a.task}_s{a.seed}", exist_ok=True, verbose=False)
    metrics = model.val(data=data, split="test")
    run_id = f"wp3_yolov8n_{a.task}_s{a.seed}"
    from src.common.db import log_metric
    log_metric(a.db, run_id, "map50", float(metrics.box.map50), wp="WP3", task=a.task,
               dataset=a.task, method="yolov8n", model="yolov8n", seed=a.seed,
               split="test", gpu_hours=(time.time() - t0) / 3600.0)
    log_metric(a.db, run_id, "map50_95", float(metrics.box.map), wp="WP3", task=a.task,
               dataset=a.task, method="yolov8n", model="yolov8n", seed=a.seed, split="test")
    print(run_id, "mAP50", float(metrics.box.map50))

if __name__ == "__main__":
    main()
