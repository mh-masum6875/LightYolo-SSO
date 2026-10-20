# WP2 Search-Method Matrix: How To Run (DICC)

Order is binding (Master Plan v2 section 6): proxy-fidelity FIRST, then the matrix.

## 0. One-time prep (local already done, verify on DICC)
- `python -m src.search.make_proxy_subset` (writes proxy_train_25pct.txt + proxy_train_paths.txt; committed, do NOT regenerate on DICC)
- `configs/loc_hard_proxy_dicc.yaml` paths match your repo location
- optuna installed in the conda env (`pip install --user optuna`)

## 1. Seed evaluations for the fidelity check
Run ONE cheap strategy first to populate the db with candidates:
    sbatch --array=0-1 slurm/wp2_array.sbatch random
Merge parts, then select the 10 fidelity candidates:
    python -c "from src.common.db import merge_parts; print(merge_parts('rebuild-v2/results/results.sqlite','rebuild-v2/results/parts/*.sqlite'))"
    python -m src.search.proxy_fidelity --select

## 2. Fully train the 10 candidates (150 epochs, full train split)
For each row in results/wp2/fidelity_candidates.csv: build the cfg from
theta_json (src.search.archgen.theta_to_cfg), train with train.py on the FULL
loc_hard train split, then log:
    python -c "from src.common.db import log_metric; log_metric('rebuild-v2/results/results.sqlite','wp2_fidelity_<i>','full_map50',<value>,wp='WP2',task='proxy_fidelity',dataset='loc_hard',tier='hard',method='full',model='theta',seed=0,split='val')"
Check:
    python -m src.search.proxy_fidelity --analyze     # needs rho >= 0.7

## 3. The matrix (only after rho >= 0.7)
    for s in sso pso random revol tpe; do sbatch slurm/wp2_array.sbatch $s; done
Each job = 100 proxy evals; array 0-9 = 10 seeds. ~800-1200 GPU-h total.

## 4. Merge + analyze (renders every table/figure from the db alone)
    python -c "from src.common.db import merge_parts; print(merge_parts('rebuild-v2/results/results.sqlite','rebuild-v2/results/parts/*.sqlite'))"
    python -m src.search.analyze

## 5. Finals, ablation (Master Plan 6.2)
Best theta per strategy -> full training x 5 seeds on loc_hard; winner also on
loc_easy, continuity, lwsndr. Stage ablation: defaults / S1 / S2 / both, 5 seeds.
