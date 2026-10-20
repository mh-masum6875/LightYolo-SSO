# WSN-FaultBench loc_easy v2.0 (built 2026-07-17, seed 0)

Fault-injection localization benchmark on IBRL clean-base windows.
Generator: src/injection/ (rebuild-v2). Bit-identical regeneration from manifest.csv proven by tests/test_injection.py.

## Construction
- Source: IBRL data.txt (2,313,727 readings, 4 modalities), rows outside plausible physical ranges dropped (temp 0-50 C, humidity 0-100 %, light 0-2000 lux, voltage 1.5-3.5 V).
- Normalization: per modality, 1st-99th percentile clip, min-max to 0-255 (params cached in base_windows.npz).
- Windows: per node, time-ordered, non-overlapping, 64 samples x 4 modalities.
- Clean-base screening: std <= 5 AND range <= 15 grayscale units per modality (half the original 10/30 labelling thresholds). Result: 14,682 clean windows from 55 nodes. 200-sample manual audit: data/benchmarks/audit_sample/.
- Injection: 45% of windows (6,607) receive 1-3 faults; no two faults share a modality; type/modality/onset/duration/intensity uniform from tier ranges (src/injection/fault_models.py TIERS). Local sigma = pre-injection std of affected modality, floor 2.0.
- Taxonomy: Sharma, Golubchik, Govindan, ACM TOSN 2010 (spike, stuck-at, drift, noise).

## Encoding
256x256 grayscale. Modality m owns rows [64m,64m+64); time step t owns cols [4t,4t+4).
Box = modality row band x [t_start,t_end], 2 px pad, clipped, min 8x8. Classes: 0 spike, 1 stuck, 2 drift, 3 noise.

## Contents
images/ (14,682 png), labels/ (YOLO txt), manifest.csv (one row per fault), {train,val,test}.txt + *_paths.txt (frozen 80/10/10, stratified faulty/clean, seed 1000).

## Stats
Faults: 13264 | drift: 3335 | noise: 3223 | spike: 3337 | stuck: 3369

## Revision history
- v2.0 (2026-07-17). Gate attempt 1 on loc_hard scored mAP@0.5 = 0.6887 (TOO HARD,
  band 0.80-0.95). Confusion matrix was class-specific: stuck recall 0.05, spike 0.69,
  drift 0.74, noise 0.96. The v1 stuck model (hold the window's own value) is
  undetectable by construction on stability-screened windows. Changes:
  (a) stuck redesigned as stuck-at-deviated-constant per Sharma et al. 2010:
      value = x[t_start] + sign*k*sigma, k hard (1.5,3), easy (4,8);
  (b) spike hard k (2,4) -> (3,5), one widening step (= Master Plan 5.1 spec);
  (c) drift hard k (1.5,3) -> (2,4), one widening step;
  (d) noise unchanged. Same build seed 0: identical window choice, fault plan
      (types/onsets/durations) and frozen splits; only intensities/semantics differ.
  Logged in results.sqlite as wp1_gate_attempt1.
- v1.0 (2026-07-14). Initial build.
