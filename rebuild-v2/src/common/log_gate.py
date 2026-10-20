"""Log a WP1 gate attempt from a YOLOv7 results.txt into results.sqlite."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.db import log_metric

def best_map50(results_txt):
    best = 0.0
    with open(results_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 11:
                try:
                    best = max(best, float(parts[10]))
                except ValueError:
                    continue
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--db", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "results.sqlite"))
    a = ap.parse_args()
    m = best_map50(a.results)
    run_id = f"wp1_gate_attempt{a.attempt}"
    log_metric(a.db, run_id, "mAP50_best", m, wp="WP1", task="gate", dataset="loc_hard",
               tier="hard", method="sso_legacy_cfg", model="wsn-yolo", seed=0, split="val",
               unit="fraction", config_path="configs/yolo_wsn_faultbench_nc4.yaml")
    verdict = "PASS" if 0.80 <= m <= 0.95 else ("TOO EASY (tighten)" if m > 0.97 else ("TOO HARD (widen)" if m < 0.70 else "BORDERLINE"))
    print(f"{run_id}: best mAP@0.5 = {m:.4f} -> {verdict}")
