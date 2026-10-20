# WP1 Gate Run - Instructions for Maidul (DICC)

Purpose: first training on loc_hard decides whether the benchmark is hard enough
to rank methods (Master Plan 5.3). WP2 does not start before this gate passes.

## Steps
1. Copy to DICC: the LightYolo-SSO repo folder AND the rebuild-v2 folder, side by side:
     ~/work/LightYolo-SSO/          (repo, has train.py, data/hyp.wsn.yaml, yolov7-tiny.pt)
     ~/work/rebuild-v2/             (this folder: configs/, data/benchmarks/loc_hard/, src/)
   loc_hard is ~14,682 pngs; zip before transfer (zip -r loc_hard.zip data/benchmarks/loc_hard).
2. rm -rf runs/train/wp1_gate* (avoid YOLOv7 auto-resume trap - known Problem A).
3. dos2unix rebuild-v2/slurm/wp1_gate.sbatch
4. cd ~/work/LightYolo-SSO && sbatch ../rebuild-v2/slurm/wp1_gate.sbatch
5. After completion, from rebuild-v2/: python -m src.common.log_gate --results ../LightYolo-SSO/runs/train/wp1_gate_loc_hard/results.txt --attempt 1
   Then send results.txt (or the printed line) back to the next Cowork session.

## Decision rule (logged, never skipped)
- mAP@0.5 in [0.80, 0.95]  -> GATE PASSES, WP2 may start.
- > 0.97 -> too easy: tighten TIERS["hard"] intensities one step in src/injection/fault_models.py
  (e.g. spike k (2,4)->(1.5,3), drift k (1.5,3)->(1,2), noise k (1.5,3)->(1,2), stuck dur (8,16)->(6,12)),
  regenerate loc_hard with the SAME seed 0, re-run. Log attempt 2.
- < 0.70 -> too hard: widen one step (opposite direction). Log attempt.
- 0.95-0.97 -> borderline: acceptable only if per-class AP spread is visible; prefer one tightening step.
