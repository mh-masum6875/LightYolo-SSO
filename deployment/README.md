# On-Device Deployment and Energy Measurement (Raspberry Pi 4)

Real-hardware evaluation of WSN-YOLO against three baselines on a Raspberry Pi 4
Model B (4 GB, 64-bit OS). Each detector is exported to ONNX and run with ONNX
Runtime (CPU execution provider). Per-inference energy is measured with a Texas
Instruments INA219 current sensor wired in series on the 5 V supply rail, sampled
during inference and aggregated by the median with the idle baseline subtracted.

## Files
- `setup.sh` — one-shot Pi setup (I2C, Python env, ONNX Runtime, base latency/RAM run).
- `measure_energy.sh` — energy run once the Pi is powered through the INA219.
- `measure_pi.py` — measurement script (latency, peak RAM, power, energy per inference).
- `WP4_HARDWARE_RUN.md` — step-by-step wiring and run notes.

## Hardware
Raspberry Pi 4 Model B · INA219 (0x40 on I2C bus 1) in series on the 5 V rail ·
sacrificial USB-A->C cable with the +5 V wire routed through Vin+/Vin-.

## Measured results (100 inferences each, idle baseline subtracted)

| Detector | ONNX size (MB) | Latency mean (ms) | Latency p95 (ms) | Peak RAM (MB) | Power (W) | Energy (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| **WSN-YOLO** | **5.5** | **197.6** | **201.5** | **123** | **4.77** | **486.9** |
| YOLOv5n | 7.9 | 321.0 | 324.5 | 128 | 4.89 | 821.2 |
| YOLOv8n | 12.8 | 490.9 | 496.7 | 130 | 5.10 | 1344.7 |
| YOLOv7-tiny | 24.2 | 575.8 | 579.9 | 166 | 5.15 | 1584.2 |

WSN-YOLO is the fastest, smallest, and lowest-energy detector on real hardware:
1.6–2.9x faster and 1.7–3.3x more energy-efficient per inference than the baselines.

## Usage
```bash
# on the Pi, from the home folder (with wp4_models.tgz + measure_pi.py present)
bash setup.sh                 # setup + base latency/RAM for all 4 models
# then power the Pi THROUGH the INA219 and run:
bash measure_energy.sh        # adds power + energy per inference
```
