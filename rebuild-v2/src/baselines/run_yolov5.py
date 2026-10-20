"""WP3 detector baseline: classic YOLOv5n (the Dependencies/yolov5 repo, run in
the yolov5_custom env), on WSN-FaultBench loc_hard detection. Mirrors how v5n was
trained in the rejected paper: yolov5n.pt weights, default hyperparameters, same
img/batch/epochs/patience as the other detectors. Trains, then evaluates on the
TEST split via the repo's val.py, and logs mAP@0.5 / mAP@0.5:0.95 to the results
DB so it lands in the same table as yolov8n.

Usage (DICC, from the yolov7_maidul root, yolov5_custom env):
  python rebuild-v2/src/baselines/run_yolov5.py --seed 0 \
     --yolov5-dir $HOME/Dependencies/yolov5 \
     --data $PWD/rebuild-v2/data/benchmarks/loc_hard/dataset_yolov5.yaml \
     --db  $PWD/rebuild-v2/results/parts/wp3_yolov5n_loc_hard_s0.sqlite
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time


def parse_val_map(stdout: str):
    """Pull mAP@0.5 and mAP@0.5:0.95 from the 'all' summary row of val.py.
    The row ends with: ... P R mAP50 mAP50-95 -> take the last two tokens."""
    map50 = map5095 = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("all "):
            toks = s.split()
            try:
                map50, map5095 = float(toks[-2]), float(toks[-1])
            except (ValueError, IndexError):
                continue
    return map50, map5095


def parse_model_info(text: str):
    """Pull parameter count and GFLOPs from a yolov5 'Model summary' line, e.g.
    'Model summary (fused): 84 layers, 1764577 parameters, 0 gradients, 4.2 GFLOPs'."""
    import re
    params = gflops = None
    for line in text.splitlines():
        if "parameters" in line and "Model summary" in line:
            m = re.search(r"([\d,]+)\s+parameters", line)
            if m:
                params = int(m.group(1).replace(",", ""))
            g = re.search(r"([\d.]+)\s*GFLOPs", line)
            if g:
                gflops = float(g.group(1))
    return params, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yolov5-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--val-only", action="store_true",
                    help="skip training; evaluate the existing best.pt on test")
    ap.add_argument("--db", required=True)
    a = ap.parse_args()

    ydir = os.path.abspath(a.yolov5_dir)
    weights = os.path.join(ydir, "yolov5n.pt")
    cfg = os.path.join(ydir, "models", "yolov5n.yaml")
    hyp = os.path.join(ydir, "data", "hyps", "hyp.scratch-low.yaml")
    project = os.path.join(ydir, "runs", "wp3_v5n")
    name = f"loc_hard_s{a.seed}"
    # PyTorch>=2.6 compat: put the sitecustomize dir on PYTHONPATH so torch.load
    # is auto-patched (weights_only=False) at interpreter startup, then run the
    # yolov5 scripts DIRECTLY so their `models` package imports resolve normally.
    compat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch26_compat")
    env = os.environ.copy()
    env["PYTHONPATH"] = compat_dir + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    best = os.path.join(project, name, "weights", "best.pt")

    if not a.val_only:
        # Mirrors the rejected-paper v5n setup (weights + cfg + scratch-low hyp,
        # SGD, batch 16, epochs 150, patience 20). Only img differs: 640 to match
        # every other detector on WSN-FaultBench (WSN-YOLO/gate, WP2, yolov8n all
        # trained at 640), so the detector comparison is on equal footing.
        train_cmd = [sys.executable, "train.py",
                     "--weights", weights, "--cfg", cfg, "--hyp", hyp,
                     "--data", a.data,
                     "--epochs", str(a.epochs), "--batch-size", str(a.batch),
                     "--imgsz", str(a.img), "--patience", "20",
                     "--seed", str(a.seed), "--project", project, "--name", name,
                     "--exist-ok", "--device", a.device]
        print("[train]", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, cwd=ydir, check=True, env=env)
    else:
        print(f"[val-only] evaluating existing weights: {best}", flush=True)
        if not os.path.exists(best):
            raise SystemExit(f"best.pt not found: {best}")

    val_cmd = [sys.executable, "val.py", "--data", a.data, "--weights", best,
               "--task", "test", "--imgsz", str(a.img),
               "--batch-size", str(a.batch), "--device", a.device]
    print("[val]", " ".join(val_cmd), flush=True)
    # yolov5 prints the results table via its LOGGER to stderr; merge both
    # streams into stdout so the parser sees the 'all' row and Model summary.
    out = subprocess.run(val_cmd, cwd=ydir, text=True, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(out.stdout)
    if out.returncode != 0:
        raise SystemExit(f"val.py failed (exit {out.returncode}); see output above")

    map50, map5095 = parse_val_map(out.stdout)
    if map50 is None:
        raise SystemExit("could not parse mAP from val.py output (see stdout above)")
    params, gflops = parse_model_info(out.stdout)

    gpu_h = (time.time() - t0) / 3600.0
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.common.db import log_metric
    rid = f"wp3_yolov5n_loc_hard_s{a.seed}"
    log_metric(a.db, rid, "map50", map50, wp="WP3", task="loc_hard",
               dataset="loc_hard", method="yolov5n", model="yolov5n",
               seed=a.seed, split="test", gpu_hours=gpu_h)
    log_metric(a.db, rid, "map50_95", map5095, wp="WP3", task="loc_hard",
               dataset="loc_hard", method="yolov5n", model="yolov5n",
               seed=a.seed, split="test")
    if params is not None:
        log_metric(a.db, rid, "params_m", params / 1e6, wp="WP3", task="loc_hard",
                   dataset="loc_hard", method="yolov5n", model="yolov5n",
                   seed=a.seed, split="test")
    if gflops is not None:
        log_metric(a.db, rid, "gflops", gflops, wp="WP3", task="loc_hard",
                   dataset="loc_hard", method="yolov5n", model="yolov5n",
                   seed=a.seed, split="test")
    print(rid, "map50", map50, "map50_95", map5095, "params_m",
          None if params is None else round(params / 1e6, 3), "gflops", gflops)


if __name__ == "__main__":
    main()
