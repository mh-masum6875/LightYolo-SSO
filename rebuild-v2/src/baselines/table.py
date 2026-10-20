"""`make baselines` target: reproduce the full WP3 table from the database alone.
Usage: python -m src.baselines.table [--db results/results.sqlite]
"""
from __future__ import annotations
import argparse, os, sqlite3, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "results", "results.sqlite"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "wp3_table.csv"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    rows = con.execute(
        "SELECT method, task, metric, seed, value FROM runs "
        "WHERE run_id LIKE 'wp3_%' AND metric IN "
        "('accuracy','f1','auroc','false_alarms_per_node_day')").fetchall()
    con.close()
    agg = defaultdict(list)
    for method, task, metric, seed, value in rows:
        agg[(method, task, metric)].append(value)
    lines = ["method,task,metric,mean,std,n_seeds"]
    for (method, task, metric), vals in sorted(agg.items()):
        lines.append(f"{method},{task},{metric},{np.mean(vals):.4f},{np.std(vals):.4f},{len(vals)}")
    with open(a.out, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {a.out}")

if __name__ == "__main__":
    main()
