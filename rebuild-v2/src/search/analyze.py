"""WP2 analysis, rendered from the database ALONE (Master Plan 6.2 acceptance).

Outputs (to results/wp2/):
  budget_check.csv        equal-budget proof: eval count per (strategy, seed)
  final_fitness.csv       J_best mean/std per strategy over seeds
  evals_to_target.csv     first eval within 1 percent of that run's best J
  wilcoxon_fitness.csv    sso vs each competitor on final fitness
  wilcoxon_evals.csv      sso vs each competitor on evaluations-to-target
  convergence.png         best-so-far curves, seed-shaded bands
  gpu_hours.csv           search cost per strategy

Usage: python -m src.search.analyze [--db results/results.sqlite]
"""
from __future__ import annotations
import argparse, os, re, sqlite3, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_RE = re.compile(r"^wp2_(?P<strategy>[a-z0-9]+)_s(?P<seed>\d+)_e(?P<idx>\d+)$")

def load_evals(db_path):
    """-> {(strategy, seed): [(idx, J), ...] sorted}, {(strategy, seed): gpu_hours}"""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT run_id, value FROM runs WHERE metric='J' AND run_id LIKE 'wp2_%_e%'").fetchall()
    hours = defaultdict(float)
    for run_id, v in con.execute(
            "SELECT run_id, value FROM runs WHERE metric='wall_s' AND run_id LIKE 'wp2_%_e%'"):
        m = RUN_RE.match(run_id)
        if m:
            hours[(m["strategy"], int(m["seed"]))] += v / 3600.0
    con.close()
    evals = defaultdict(list)
    for run_id, J in rows:
        m = RUN_RE.match(run_id)
        if m:
            evals[(m["strategy"], int(m["seed"]))].append((int(m["idx"]), float(J)))
    return {k: sorted(v) for k, v in evals.items()}, dict(hours)

def best_so_far(js):
    out, b = [], float("inf")
    for j in js:
        b = min(b, j)
        out.append(b)
    return out

def evals_to_target(js, tol=0.01):
    """First eval index (1-based) within tol of THIS run's best-found fitness."""
    best = min(js)
    target = best + tol * abs(best)
    for i, j in enumerate(js):
        if j <= target:
            return i + 1
    return len(js)

def wilcoxon_table(per_strategy, reference="sso"):
    from scipy.stats import ranksums
    rows = []
    ref = per_strategy.get(reference, [])
    for s, vals in sorted(per_strategy.items()):
        if s == reference or not ref:
            continue
        stat, p = ranksums(ref, vals)
        rows.append((reference, s, float(np.mean(ref)), float(np.mean(vals)), float(stat), float(p)))
    return rows

def write_csv(path, header, rows):
    with open(path, "w", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "results", "results.sqlite"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "wp2"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    evals, hours = load_evals(a.db)
    if not evals:
        print("no WP2 evaluations in database"); return

    write_csv(os.path.join(a.out, "budget_check.csv"),
              ["strategy", "seed", "n_evals"],
              [(s, sd, len(v)) for (s, sd), v in sorted(evals.items())])

    finals, e2t = defaultdict(list), defaultdict(list)
    for (s, sd), v in evals.items():
        js = [j for _, j in v]
        finals[s].append(min(js))
        e2t[s].append(evals_to_target(js))
    write_csv(os.path.join(a.out, "final_fitness.csv"),
              ["strategy", "mean_J_best", "std_J_best", "n_seeds"],
              [(s, np.mean(v), np.std(v), len(v)) for s, v in sorted(finals.items())])
    write_csv(os.path.join(a.out, "evals_to_target.csv"),
              ["strategy", "mean", "std", "n_seeds"],
              [(s, np.mean(v), np.std(v), len(v)) for s, v in sorted(e2t.items())])
    write_csv(os.path.join(a.out, "gpu_hours.csv"),
              ["strategy", "seed", "gpu_hours"],
              [(s, sd, round(h, 3)) for (s, sd), h in sorted(hours.items())])
    try:
        write_csv(os.path.join(a.out, "wilcoxon_fitness.csv"),
                  ["ref", "vs", "mean_ref", "mean_vs", "stat", "p"], wilcoxon_table(finals))
        write_csv(os.path.join(a.out, "wilcoxon_evals.csv"),
                  ["ref", "vs", "mean_ref", "mean_vs", "stat", "p"], wilcoxon_table(e2t))
    except ImportError:
        print("scipy missing: Wilcoxon tables skipped")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in sorted({k[0] for k in evals}):
            curves = [best_so_far([j for _, j in evals[(s2, sd)]])
                      for (s2, sd) in evals if s2 == s]
            L = min(len(c) for c in curves)
            arr = np.array([c[:L] for c in curves])
            x = np.arange(1, L + 1)
            ax.plot(x, arr.mean(0), label=f"{s} (n={len(curves)})")
            ax.fill_between(x, arr.min(0), arr.max(0), alpha=0.15)
        ax.set_xlabel("evaluation"); ax.set_ylabel("best-so-far J (lower is better)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "convergence.png"), dpi=200)
        print("convergence.png written")
    except ImportError:
        print("matplotlib missing: convergence figure skipped")
    print(f"analysis written to {a.out}")

if __name__ == "__main__":
    main()
