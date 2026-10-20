# WP3 cross-family baselines — run sheet

Answers reviewer R2.4 (compare against methods outside the YOLO family) and R4
(off-the-shelf modern detector), all judged on the **same** WSN-FaultBench that
WSN-YOLO is evaluated on. One submit runs everything; caps at the 2 idle GPUs.

## What runs (40 jobs, `slurm/wp3_jobs.txt`)

- **Raw-stream baselines on the reconstructed fault-stream** (same faulted 64x4
  windows WSN-YOLO sees as images): `lstm_ae`, `dense_ae`, `cnn1d`, `gru_clf`,
  `iforest` (5 seeds each), `ocsvm`, `rule` (1 seed). No `data.txt` needed — the
  stream is rebuilt bit-identically from `base_windows.npz` + the loc_hard
  manifest.
- **Image classifiers on loc_hard binary**: `mobilenetv3s`, `shufflenet`
  (5 seeds) — shows what classification cannot localise.
- **Off-the-shelf detector on loc_hard detection**: `yolov8n` (3 seeds) — the R4
  reference against our SSO-tuned detector.

Local check already done: on the fault-stream, `iforest` AUROC 0.82, `ocsvm`
0.74, `rule` 0.50 (the expected degenerate floor). The learned torch models see
the same signal.

## Steps (from the `yolov7_maidul` root, where you launched the gate/WP2)

1. Drop these files in (preserving paths): `rebuild-v2/src/baselines/data.py`,
   `rebuild-v2/src/baselines/run_baseline.py`, `rebuild-v2/slurm/wp3_run.sbatch`,
   `rebuild-v2/slurm/wp3_jobs.txt`, `rebuild-v2/slurm/wp3_collect.py`.

2. One-time: cache the yolov8n weights on the login node so the compute nodes
   never need internet (only the 3 `yolo` jobs depend on this):
   ```
   conda activate wp3
   cd rebuild-v2
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"   # downloads yolov8n.pt into cwd
   ```
   If this login-node download fails, submit anyway — the 37 non-yolo jobs (the
   full R2.4 comparison) are self-contained and unaffected.

3. Submit (one command):
   ```
   sbatch slurm/wp3_run.sbatch
   ```
   Watch: `squeue -u $USER`. Logs: `logs/wp3_<jobid>_<idx>.out`. Each raw/image
   job is minutes; the 3 yolov8n jobs are the long pole (~150 epochs).

4. When the array finishes, collect the table:
   ```
   cd rebuild-v2 && python slurm/wp3_collect.py
   ```
   Prints mean+-std over seeds and writes `results/wp3_table.csv`.

## Notes

- Everything writes to `results/parts/wp3_*.sqlite` (one file per job, no write
  contention); step 4 folds them into `results/results.sqlite`.
- Uses the `wp3` conda env (sklearn + ultralytics + torch), not `yolov7_maidul`.
- **lwsndr transfer baselines are intentionally not in this wave**: their raw
  64x4 streams are not on DICC (only IBRL's `base_windows.npz` is) and the
  lwsndr manifest stores labels only, so the raw-stream models cannot be rebuilt
  self-contained. This wave gives the complete cross-family comparison on
  WSN-FaultBench; lwsndr can follow once its raw streams are confirmed.
