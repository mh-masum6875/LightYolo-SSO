# LightYolo-SSO (WSN-YOLO)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

An ultra-lightweight object detector for wireless sensor networks. Its architecture is found with
Sperm Swarm Optimization (SSO) instead of by shrinking an existing model, so it suits sensor nodes
with very little memory and power. It builds on [YOLOv7](https://github.com/WongKinYiu/yolov7).
The manuscript is in preparation.

## Approach

SSO searches the network structure directly. Each candidate is scored on accuracy together with
FLOPs, parameter count and inference time. WSN-YOLO is compared with YOLOv5n, YOLOv8n and YOLOv7-tiny
under the same data and training conditions, and then measured on a Raspberry Pi 4 with an INA219
power sensor.

## Results

### Fault localisation on WSN-FaultBench (loc_hard, 640 px)

| Detector | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|
| **WSN-YOLO** | 0.913 | 0.832 | **1.35** | **2.6** |
| YOLOv5n | 0.925 | 0.857 | 1.76 | 4.2 |
| YOLOv8n | 0.945 | 0.896 | 3.01 | 8.1 |
| YOLOv7-tiny | 0.930 | 0.865 | 6.02 | 13.2 |

WSN-YOLO sits at the efficiency knee: within one to three points of much larger detectors, with
4.5 times fewer parameters than YOLOv7-tiny.

### On a Raspberry Pi 4 (ONNX Runtime, CPU, 100 inferences each)

| Detector | ONNX size (MB) | Latency mean (ms) | Peak RAM (MB) | Power (W) | Energy (mJ) |
|---|---:|---:|---:|---:|---:|
| **WSN-YOLO** | **5.5** | **197.6** | **123** | **4.77** | **486.9** |
| YOLOv5n | 7.9 | 321.0 | 128 | 4.89 | 821.2 |
| YOLOv8n | 12.8 | 490.9 | 130 | 5.10 | 1344.7 |
| YOLOv7-tiny | 24.2 | 575.8 | 166 | 5.15 | 1584.2 |

Energy per inference comes from an inline INA219 sensor with the idle baseline subtracted. WSN-YOLO
runs 1.6 to 2.9 times faster and uses 1.7 to 3.3 times less energy than the baselines. Wiring and
protocol are in [`deployment/`](deployment/).

## Where things are

```
.
├── rebuild-v2/        The current research pipeline (start here to reproduce results)
│   ├── src/           benchmark builder, SSO search, baselines, statistics
│   ├── configs/       loc_hard and search configs, WSN-YOLO model config
│   ├── slurm/         batch scripts for each work package
│   ├── data/          benchmark definitions (raw images are not committed)
│   ├── results/       results.sqlite and YOLOv8n curves
│   └── trained_models/  weights that are included (see the table below)
├── deployment/        Raspberry Pi 4 and INA219 latency and energy measurement
├── SSO/               SSO optimisers in Python, plus the original MATLAB version
├── cfg/, models/, utils/, data/   YOLOv7 code base this project extends
├── train.py, test.py, detect.py, export.py   YOLOv7 scripts
└── tools/             export notebooks inherited from the YOLOv7 repository
```

The scripts at the top level and `data/wsn_dataset.yaml` come from the earlier two-class
(normal and anomaly) setup. The current benchmark has four fault classes (spike, stuck, drift, noise),
and its configs live in `rebuild-v2/`.

## Reproduce the loc_hard training

The benchmark is regenerated from the Intel Berkeley Research Laboratory streams with
`rebuild-v2/src/injection/build_benchmark.py`. Split sizes for loc_hard are 11,745 train, 1,467
validation and 1,470 test windows of 256 by 256 pixels. Then train WSN-YOLO:

```bash
pip install -r requirements.txt
python train.py \
  --weights yolov7-tiny.pt \
  --cfg rebuild-v2/configs/yolo_wsn_faultbench_nc4.yaml \
  --data rebuild-v2/configs/loc_hard_dicc.yaml \
  --hyp data/hyp.wsn.yaml \
  --img-size 640 640 --batch-size 16 --epochs 150 \
  --name wsn_yolo_loc_hard
```

Edit the paths inside `loc_hard_dicc.yaml` to point at your copy of the benchmark. The YOLOv7-tiny
weights come from the [YOLOv7 releases](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1).
The full set of runs, in order, is described in `rebuild-v2/README.md` and the `*_RUN.md` files.

## Search

```bash
cd SSO
python sso_architecture_optimizer.py --population 20 --iterations 30
python sso_hyperparameter_optimizer.py --population 15 --iterations 20
```

The multi-objective fitness is `J = w1(1 - mAP) + w2 FLOPs/F_target + w3 Params/P_target + w4 T_inf`
with weights 0.4, 0.3, 0.2 and 0.1. The velocity update follows the original SSO paper (see References).

## Trained weights included

| File | What it is |
|---|---|
| `rebuild-v2/trained_models/loc_hard_yolov8n.pt`, `yolov8n.onnx` | YOLOv8n on loc_hard, and the ONNX used for the Pi test |
| `loc_hard_gate_yolov7tiny.pt` | the detector used as the benchmark gate |
| `loc_lwsndr_wsnyolo_scratch.pt`, `_transfer.pt` | WSN-YOLO on the second benchmark (LWSNDR) |
| `loc_lwsndr_yolov7tiny_scratch.pt`, `_transfer.pt` | YOLOv7-tiny on the second benchmark |

The YOLOv8n files are Ultralytics checkpoints. Use them with the Ultralytics tools, for example
`yolo predict model=rebuild-v2/trained_models/loc_hard_yolov8n.pt source=path/to/images`, not with
this repository's `detect.py`.

## Citation

The paper is in preparation. Until it is out, please cite this repository.

## References

1. Shehadeh, H.A., et al. "Sperm swarm optimization algorithm for optimizing wireless sensor network challenges." IEEE Access 9 (2021).
2. Wang, C.Y., et al. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." CVPR 2023.

The SSO code is based on the original implementation by Shehadeh et al. The detector code is based on [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

## License

GPL-3.0, inherited from YOLOv7. See [LICENSE](LICENSE).
