# Legacy Log Manifest (WP0) — updated 2026-07-14 after exploring F:\Masters\Paper 3\Model

| Artifact | Location | Status |
|---|---|---|
| WSN-YOLO final training (150 ep, mAP50 0.9957) | GitHub/LightYolo-SSO/runs/train/WSN_YOLO/results.txt + tfevents + weights | present |
| YOLOv7-tiny baseline (150 ep, mAP50 0.9969) | Model/11thAUg2025/done_tiny_TRain/results.txt + tfevents | present |
| Collapsed early run (mAP->0 at ep ~79, box=0.05 bug) | Model/yolov7_maidul/results.txt | present, documents Problem B |
| Many intermediate runs (wsn_sso_transfer1-8, wsn_sso_optimized*) | Model/yolov7_maidul/runs/ | present, mostly empty/partial |
| Stage 1 SSO final output | Model/11thAUg2025/SSO/optimized_architectures/{yolo_wsn_optimized.yaml, optimization_metrics.yaml} | present (final vector only; metrics yaml has placeholder values, flops_estimate 0.0) |
| Stage 2 SSO final output | Model/yolov7_maidul/data/hyp.wsn.sso_optimized.yaml, hyp.wsn.yaml | present (final hyp only) |
| Per-iteration SSO fitness logs (Stage 1 & 2) | — | **DO NOT EXIST.** Optimizer code (sso_yolo_optimizer.py, sso_*_optimizer.py) never logged iteration history; only final yaml dumps. DICC search unnecessary. |

Conclusion: legacy search evidence is limited to final vectors. All convergence/variance
claims in the new paper must come from WP2 reruns. Manuscript claim "pop 20, 30 iterations"
for Stage 1 cannot be verified from logs; report WP2 protocol instead.
