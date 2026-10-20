# rebuild-v2 — current WSN-YOLO research pipeline

This folder is the pipeline behind the IEEE Sensors Journal submission: the
WSN-FaultBench fault-localisation benchmark, the dual-stage Enhanced-SSO
architecture search, the cross-family baselines, the statistical analysis, and
the on-device deployment. It supersedes the earlier binary-classification setup.

## Layout
- `src/injection/` — WSN-FaultBench builder (fault injection + image/box encoding) and the LWSNDR second benchmark.
- `src/search/`    — dual-stage SSO architecture + hyperparameter search (archgen, fitness, analyze).
- `src/baselines/` — cross-family baselines (TCN, GRU, autoencoders, one-class SVM, isolation forest, threshold rule, image classifiers).
- `src/stats/`, `src/common/`, `src/deploy/` — statistics, DB/metrics/seeds helpers, deployment glue.
- `configs/`       — loc_hard configs and the per-optimiser search configs (SSO, PSO, random, reg-evolution).
- `slurm/`         — DICC batch scripts for every work package.
- `data/benchmarks/` — benchmark **definitions** (cards, dataset.yaml, manifests, split path-lists, base_windows.npz). Raw images/labels are not committed; regenerate them with `src/injection/build_benchmark.py`.
- `results/results.sqlite` — all logged metrics (single `runs` table, ~36k rows): frontier, optimiser contest, detector comparison, cross-family, generalisation, deployment.
- `results/wp3_yolov8/` — YOLOv8n loc_hard runs with PR / confusion / results curves.
- `trained_models/` — available trained weights and the YOLOv8n ONNX (see below).

## Trained models in this folder
| File | What it is |
|---|---|
| `loc_hard_yolov8n.pt`, `yolov8n.onnx` | YOLOv8n trained on loc_hard (+ONNX used for the Pi test) |
| `loc_hard_gate_yolov7tiny.pt` | the loc_hard benchmark gate detector |
| `loc_lwsndr_wsnyolo_scratch.pt`, `_transfer.pt` | WSN-YOLO on the LWSNDR second dataset |
| `loc_lwsndr_yolov7tiny_scratch.pt`, `_transfer.pt` | YOLOv7-tiny on the LWSNDR second dataset |
| `yolo26n.pt` | baseline weight (confirm the exact version label) |

Not yet included (large, in the DICC `runs/` tree): the **WSN-YOLO loc_hard primary
weights** and the **WSN-YOLO / YOLOv5n / YOLOv7-tiny ONNX** used for the Pi
deployment. These can be added on request.

## Headline result (loc_hard, 640 px)
WSN-YOLO: 1.35 M params, 2.6 GFLOPs, 0.913 mAP@0.5 — the efficiency knee; lowest
latency, memory and measured per-inference energy of four detectors on a
Raspberry Pi 4 (see `../deployment/`).
