# WP2 run sheet — SSO optimizer contest (reviewer R2.3)

Isolates SSO's contribution: SSO vs PSO, random search, regularized evolution,
and TPE (Bayesian), all on the identical 15-D space at equal budget, 10 seeds
each, with convergence, evals-to-target, variance, Wilcoxon, and search cost.

**Efficiency framing.** Your WP1 frontier proved mAP@0.5 is flat (0.905–0.930)
across the whole feasible space, and the 20-epoch accuracy proxy was dead
(ρ=0.048). So the search minimizes the paper's exact J with the cost terms
measured for real (thop) and accuracy from a surrogate calibrated to the 9
frontier points. J then has an interior optimum at the efficiency knee, so the
optimizers compete on how efficiently they find it. This runs on CPU in minutes
— no training, no GPU needed for the search itself.

## Step 0 — smoke test on the login node (30 s, do this first)

```
conda activate yolov7_maidul
cd ~/yolov7_maidul
python -m pip show thop optuna >/dev/null 2>&1 || pip install --user thop optuna
export PYTHONPATH="$PWD/rebuild-v2:$PWD"
python -m src.search.run --strategy sso --seed 0 --budget 5 --backend cost \
  --yolov7-dir "$PWD" --base-cfg "$PWD/cfg/training/yolov7-tiny.yaml" --img 640 \
  --db /tmp/wp2_smoke.sqlite
```
Expect a line like `wp2_sso_s0: 5 evals, J_best = ...`. That confirms CostBackend
builds real architectures and measures params/FLOPs. If `cfg/training/yolov7-tiny.yaml`
isn't the right base cfg path, adjust `--base-cfg` (use the same yolov7-tiny cfg
the gate/frontier used) and rerun.

## Step 1 — submit the contest (50 jobs = 5 strategies × 10 seeds)

From the yolov7_maidul root:
```
sbatch rebuild-v2/slurm/wp2_cost.sbatch          # default budget 100
# or a different budget:  sbatch rebuild-v2/slurm/wp2_cost.sbatch 150
```
Each job ~5 min (100 CPU evals). `%3` caps concurrency. Watch `squeue -u $USER`;
logs in `logs/wp2_<jobid>_<idx>.out`.

## Step 2 — collect + render the R2.3 analysis

```
cd ~/yolov7_maidul/rebuild-v2
python -m pip show scipy matplotlib >/dev/null 2>&1 || pip install --user scipy matplotlib
python slurm/wp2_collect.py
```
Writes to `results/wp2/`: `budget_check.csv` (equal-budget proof),
`final_fitness.csv` (mean±std J per strategy), `evals_to_target.csv`,
`wilcoxon_fitness.csv` + `wilcoxon_evals.csv` (SSO vs each competitor),
`convergence.png` (best-so-far curves), `gpu_hours.csv` (search cost).

## Step 3 — validate the winner (one GPU training, optional but recommended)

The SSO best theta is logged as `wp2_sso_s*` (config_path column). It should land
near the frontier knee (~1.35M params). Confirm its accuracy is in the band by
training it once at full fidelity — or simply cite frontier arch_07 (0.913
mAP@0.5), which is the same operating point.

## Notes

- Backend is `cost` (CPU thop), not `yolo` (GPU proxy). The old proxy path still
  exists but is not used — it was dead (ρ=0.048).
- Grid search is intractable in 15-D (even 2 points/dim = 32768 evals); random
  search is the standard uninformed baseline and is included.
