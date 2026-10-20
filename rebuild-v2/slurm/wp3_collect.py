"""Fold the per-job WP3 part databases into the master and print the R2.4 table.
Run on the login node AFTER the array finishes:
    conda activate wp3
    cd rebuild-v2 && python slurm/wp3_collect.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.db import merge_parts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

master = os.path.join(ROOT, "results", "results.sqlite")
n = merge_parts(master, os.path.join(ROOT, "results", "parts", "wp3_*.sqlite"))
print(f"merged {n} metric rows from results/parts/ into {master}\n")
os.system(f'{sys.executable} -m src.baselines.table --db "{master}"')
