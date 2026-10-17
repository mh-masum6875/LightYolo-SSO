# WSN-YOLO: Ultra-Lightweight Object Detection for Wireless Sensor Networks via Sperm Swarm Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ultra-lightweight YOLO architecture optimized for deployment in resource-constrained Wireless Sensor Networks (WSN) using Sperm Swarm Optimization (SSO) for neural architecture search. Based on [YOLOv7](https://github.com/WongKinYiu/yolov7).

## Approach

The architecture is not a shrunk copy of an existing model. Sperm Swarm Optimization (SSO) searches the network structure directly, scoring each candidate on accuracy together with FLOPs, parameter count and inference time, so the result suits wireless sensor nodes with very little memory and power. It is compared against YOLOv7-tiny and YOLOv5n baselines under the same data and training conditions. Detailed results are being updated as the work continues, so figures are not listed here.

## Project Structure

```
yolov7/
├── cfg/                        # Model configurations
│   ├── training/
│   │   ├── yolo_wsn_sso_optimized.yaml  # SSO-optimized WSN model
│   │   └── yolov7-tiny.yaml             # Baseline model
│   └── deploy/
├── data/                       # Dataset configs & hyperparameters
│   ├── hyp.wsn.yaml           # WSN-optimized hyperparameters
│   ├── hyp.scratch.tiny.yaml  # Baseline hyperparameters
│   └── wsn_dataset.yaml       # Dataset configuration
├── models/                     # Model implementations
├── utils/                      # Utility functions
├── SSO/                        # SSO optimization code
│   ├── sso_architecture_optimizer.py
│   ├── sso_hyperparameter_optimizer.py
│   └── SSO.m                   # Original MATLAB implementation
├── runs/                       # Training outputs
│   └── train/WSN_YOLO/        # Trained WSN-YOLO results
├── train.py                    # Training script
├── test.py                     # Validation script
├── detect.py                   # Inference script
└── export.py                   # Model export script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mh-masum6875/LightYolo-SSO
cd LightYolo-SSO

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The WSN-YOLO model is trained on the Intel Berkeley Research Laboratory (IBRL) dataset:
- **Total Images**: 31,067 (256×256 grayscale)
- **Classes**: 2 (Normal, Anomaly)
- **Split**: Train (24,853) / Val (3,105) / Test (3,109)

## Training

### Train WSN-YOLO (SSO-optimized)

```bash
python train.py \
    --weights yolov7-tiny.pt \
    --cfg cfg/training/yolo_wsn_sso_optimized.yaml \
    --data data/wsn_dataset.yaml \
    --hyp data/hyp.wsn.yaml \
    --epochs 150 \
    --batch-size 32 \
    --img-size 640 \
    --name wsn_yolo
```

### Train YOLOv7-tiny baseline

```bash
python train.py \
    --weights yolov7-tiny.pt \
    --cfg cfg/training/yolov7-tiny.yaml \
    --data data/wsn_dataset.yaml \
    --hyp data/hyp.scratch.tiny.yaml \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640 \
    --name yolov7_tiny_baseline
```

## SSO Optimization

### Architecture Search

```bash
cd SSO
python sso_architecture_optimizer.py --population 20 --iterations 30
```

### Hyperparameter Optimization

```bash
python sso_hyperparameter_optimizer.py --population 15 --iterations 20
```

## Inference

```bash
python detect.py \
    --weights runs/train/WSN_YOLO/weights/best.pt \
    --source path/to/images \
    --img-size 640 \
    --conf-thres 0.25
```

## Validation

```bash
python test.py \
    --data data/wsn_dataset.yaml \
    --weights runs/train/WSN_YOLO/weights/best.pt \
    --task val
```

## Pre-trained Weights

Download YOLOv7-tiny pre-trained weights:
```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

## SSO Algorithm

The Sperm Swarm Optimization algorithm uses biologically-inspired movement patterns:

**Velocity Update:**
```
v(t+1) = r₁·ln(ρ₁)·v(t) + ln(ρ₂)·ln(ρ₃)·(pbest - x(t)) + ln(ρ₂)·ln(ρ₃)·(gbest - x(t))
```

**Multi-Objective Fitness:**
```
J(θ) = w₁·(1 - mAP) + w₂·FLOPs/F_tar + w₃·Params/P_tar + w₄·T_inf
```

With weights [0.4, 0.3, 0.2, 0.1] balancing accuracy and efficiency.

## Training Results

The trained WSN-YOLO model results are available in `runs/train/WSN_YOLO/`:
- `weights/best.pt` - Best performing model
- `results.txt` - Training metrics per epoch
- `confusion_matrix.png` - Confusion matrix visualization
- `F1_curve.png`, `PR_curve.png` - Performance curves

## Citation

```bibtex
@article{wsn_yolo_sso,
  title={Ultra-Lightweight YOLO Architecture for Wireless Sensor Networks via Sperm Swarm Optimization},
  author={Maidul Hasan Masum},
  year={2025}
}
```

## References

1. Shehadeh, H.A., et al. "Sperm swarm optimization algorithm for optimizing wireless sensor network challenges." IEEE Access 9 (2021).
2. Wang, C.Y., et al. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." CVPR 2023.

## License

This project is licensed under the MIT License.

## Acknowledgments

- YOLOv7 implementation based on [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- SSO algorithm based on Shehadeh et al.'s original implementation

