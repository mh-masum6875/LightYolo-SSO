# WP1 Gate — ATTEMPT 2 (after benchmark revision 2, 2026-07-17)

Attempt 1 scored mAP@0.5 = 0.6887 (TOO HARD). Diagnosis: stuck faults were
invisible by construction (recall 0.05). The benchmark was revised (v2.0):
stuck = stuck-at-deviated-constant; spike/drift hard ranges widened one step.
Details: data/benchmarks/loc_hard/benchmark_card.md, Revision history.

## What changed on disk (re-upload to DICC)
Only these need to be re-uploaded (do NOT re-upload the whole folder):
1. `rebuild-v2/src/injection/fault_models.py`  (revised fault models)
2. `rebuild-v2/data/benchmarks/loc_hard/images/` + `labels/` + `manifest.csv` (regenerated)
   OR, faster if base_windows.npz is already on DICC:
   delete old images/labels on DICC, upload only fault_models.py, then run
   `python -m src.injection.build_benchmark --tier hard --seed 0`
   (bit-identical to the local build; splits files unchanged, no re-upload needed).

## Run (identical to attempt 1)
    cd /home/user/mhmasum6875/yolov7_maidul
    sed -i 's/\r$//' train_gpu.sh        # if edited on Windows
    sbatch train_gpu.sh

## Decision rule (same as before)
Read the last lines of the newest job_*.out:
- PASS  0.80–0.95  -> gate closed; log with
  `python -m src.common.log_gate --results <results.txt> --attempt 2`
  then START WP2: see configs/WP2_RUN_README.md (proxy-fidelity FIRST).
- >0.97 TOO EASY  -> tighten one step, regenerate, attempt 3.
- <0.70 TOO HARD  -> check the confusion matrix FIRST (attempt 1 lesson:
  look per-class before widening blindly), then adjust and regenerate.
