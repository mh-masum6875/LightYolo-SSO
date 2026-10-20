"""Fold the per-job WP2 part databases into the master and render the R2.3
analysis (equal-budget proof, final-fitness, evals-to-target, Wilcoxon vs SSO,
convergence figure, search-cost). Run on the login node after the array:
    conda activate yolov7_maidul   # or any env with scipy + matplotlib
    cd rebuild-v2 && python slurm/wp2_collect.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.db import merge_parts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

master = os.path.join(ROOT, "results", "results.sqlite")
n = merge_parts(master, os.path.join(ROOT, "results", "parts", "wp2_*.sqlite"))
print(f"merged {n} WP2 metric rows into {master}\n")
rc = os.system(f'{sys.executable} -m src.search.analyze --db "{master}"')
print("\n(analyze.py exit:", rc, ") -> CSVs + convergence.png in results/wp2/")
print("If Wilcoxon/figure were skipped: pip install --user scipy matplotlib")
