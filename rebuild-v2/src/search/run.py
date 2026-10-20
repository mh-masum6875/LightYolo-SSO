"""WP2 driver: one (strategy, seed) search run of exactly --budget evaluations.

SLURM-array friendly (Master Plan 11): writes to results/parts/
wp2_{strategy}_s{seed}.sqlite; merge with src.common.db.merge_parts afterwards.

Usage:
  python -m src.search.run --strategy sso --seed 0 --budget 100 --backend yolo \
      --yolov7-dir . --data configs/loc_hard_proxy_dicc.yaml --hyp hyp.wsn.yaml
  python -m src.search.run --strategy random --seed 0 --backend mock   # local dry run
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.search import space
from src.search.optimizers import make, STRATEGIES
from src.search.fitness import BACKENDS, FitnessLogger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_search(strategy: str, seed: int, budget: int, fitness: FitnessLogger):
    opt = make(strategy, seed)
    best_theta, best_J = None, float("inf")
    while fitness.n_evals < budget:
        thetas = opt.ask()
        room = budget - fitness.n_evals
        truncated = len(thetas) > room
        thetas = thetas[:room]
        fitnesses = [fitness(t) for t in thetas]
        for t, J in zip(thetas, fitnesses):
            if J < best_J:
                best_theta, best_J = np.asarray(t, dtype=float).copy(), float(J)
        if not truncated:
            opt.tell(thetas, fitnesses)   # partial batches end the run; no lopsided update
    return best_theta, best_J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="yolo")
    ap.add_argument("--db")
    ap.add_argument("--yolov7-dir", default=".")
    ap.add_argument("--base-cfg", help="base yolov7-tiny cfg for archgen (cost backend)")
    ap.add_argument("--data")
    ap.add_argument("--hyp")
    ap.add_argument("--proxy-epochs", type=int, default=20)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default="yolov7-tiny.pt",
                    help="transfer weights; pass '' to train from scratch")
    a = ap.parse_args()

    db = a.db or os.path.join(ROOT, "results", "parts", f"wp2_{a.strategy}_s{a.seed}.sqlite")
    if a.backend == "mock":
        backend = BACKENDS["mock"]()
    elif a.backend == "cost":
        # CPU-cheap efficiency contest: real params/FLOPs via thop, no training.
        backend = BACKENDS["cost"](a.yolov7_dir, base_cfg=a.base_cfg, img=a.img)
    else:
        if not (a.data and a.hyp):
            ap.error("--data and --hyp are required with --backend yolo")
        backend = BACKENDS["yolo"](a.yolov7_dir, a.data, a.hyp,
                                   proxy_epochs=a.proxy_epochs, img=a.img,
                                   batch=a.batch, device=a.device, weights=a.weights)
    fitness = FitnessLogger(backend, db, a.strategy, a.seed)
    t0 = time.time()
    theta, J = run_search(a.strategy, a.seed, a.budget, fitness)
    from src.common.db import log_metric
    log_metric(db, f"wp2_{a.strategy}_s{a.seed}", "J_best", float(J), wp="WP2",
               task="search_summary", dataset="loc_hard", tier="hard",
               method=a.strategy, model="theta", seed=a.seed, split="val",
               config_path=space.theta_to_json(space.discretize(theta)),
               gpu_hours=(time.time() - t0) / 3600.0)
    print(f"wp2_{a.strategy}_s{a.seed}: {fitness.n_evals} evals, "
          f"J_best = {J:.6f}, theta = {space.theta_to_json(space.discretize(theta))}")

if __name__ == "__main__":
    main()
