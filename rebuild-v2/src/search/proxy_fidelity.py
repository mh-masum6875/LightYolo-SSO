"""R2 proxy-fidelity check - RUNS FIRST, gates the rest of WP2.

Step 1 (--select): after early search runs exist, pick 10 evaluated
architectures spanning the proxy-J range (deciles of J), write
results/wp2/fidelity_candidates.csv with run_id + theta json.
Step 2 (manual/DICC): train each candidate FULLY (150 epochs, full train
split), log full mAP@0.5 to the db as run_id wp2_fidelity_{i} metric full_map50.
Step 3 (--analyze): Spearman rank correlation proxy vs full. PASS if rho >= 0.7,
else raise proxy epochs to 40 and re-check (Master Plan 6.2).

Usage:
  python -m src.search.proxy_fidelity --select
  python -m src.search.proxy_fidelity --analyze
"""
from __future__ import annotations
import argparse, os, sqlite3, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.path.join(ROOT, "results", "results.sqlite")
OUT = os.path.join(ROOT, "results", "wp2")

def select(db_path, n=10):
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT run_id, value, config_path FROM runs "
        "WHERE metric='J' AND run_id LIKE 'wp2_%_e%' ORDER BY value").fetchall()
    con.close()
    if len(rows) < n:
        print(f"only {len(rows)} evaluations available, need >= {n}"); return
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fidelity_candidates.csv")
    with open(path, "w", newline="\n") as f:
        f.write("fidelity_id,source_run_id,proxy_J,theta_json\n")
        for i, k in enumerate(idx):
            run_id, J, theta = rows[k]
            f.write(f'{i},{run_id},{J},"{theta}"\n')
    print(f"{n} candidates spanning J [{rows[idx[0]][1]:.4f}, {rows[idx[-1]][1]:.4f}] -> {path}")

def analyze(db_path):
    from scipy.stats import spearmanr
    import csv
    cand = {}
    with open(os.path.join(OUT, "fidelity_candidates.csv")) as f:
        for r in csv.DictReader(f):
            cand[int(r["fidelity_id"])] = float(r["proxy_J"])
    con = sqlite3.connect(db_path)
    full = dict(con.execute(
        "SELECT run_id, value FROM runs WHERE metric='full_map50' AND run_id LIKE 'wp2_fidelity_%'").fetchall())
    con.close()
    pairs = []
    for i, pj in sorted(cand.items()):
        rid = f"wp2_fidelity_{i}"
        if rid in full:
            pairs.append((-pj, full[rid]))   # -J so both increase with quality
    if len(pairs) < len(cand):
        print(f"WARNING: only {len(pairs)}/{len(cand)} full trainings logged")
    if len(pairs) < 3:
        print("not enough pairs"); return
    rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    verdict = "PASS (matrix may continue)" if rho >= 0.7 else "FAIL (raise proxy epochs to 40, re-check)"
    print(f"Spearman rho = {rho:.3f} (p = {p:.4f}) over {len(pairs)} pairs -> {verdict}")
    from src.common.db import log_metric
    log_metric(db_path, "wp2_proxy_fidelity", "spearman_rho", float(rho), wp="WP2",
               task="proxy_fidelity", dataset="loc_hard", tier="hard",
               method="all", model="theta", split="val")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--db", default=DB)
    a = ap.parse_args()
    if a.select: select(a.db)
    elif a.analyze: analyze(a.db)
    else: print("use --select or --analyze")

if __name__ == "__main__":
    main()
