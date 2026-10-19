# WP4 — on-device deployment measurement (Raspberry Pi)

Produces the deployment table for reviewer R2.2: model size, latency, RAM, and
energy per inference for WSN-YOLO and the baseline detectors, measured on a real
Raspberry Pi. Energy uses the INA219 in the Pi's power line.

---

## Stage 1 — export the 4 detectors to ONNX (on DICC)

ONNX is the portable format the Pi runs. Latency/size depend on the architecture,
so export each trained detector once. From the yolov7_maidul root:

```
conda activate yolov7_maidul
# WSN-YOLO (arch_07) and the YOLOv7-tiny-class full-width model (arch_05)
python export.py --weights runs/frontier/arch_07/weights/best.pt --img-size 640 640 --simplify --grid
mv runs/frontier/arch_07/weights/best.onnx wsnyolo.onnx
python export.py --weights runs/frontier/arch_05/weights/best.pt --img-size 640 640 --simplify --grid
mv runs/frontier/arch_05/weights/best.onnx yolov7tiny.onnx

# YOLOv5n (classic repo)
conda activate yolov5_custom
python ~/Dependencies/yolov5/export.py --weights ~/Dependencies/yolov5/yolov5n.pt --include onnx --imgsz 640
mv ~/Dependencies/yolov5/yolov5n.onnx ~/yolov7_maidul/yolov5n.onnx

# YOLOv8n (ultralytics)
conda activate wp3
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
mv yolov8n.onnx ~/yolov7_maidul/yolov8n.onnx
```

Grab one sample image too:
```
cp rebuild-v2/data/benchmarks/loc_hard/images/w000000.png ~/yolov7_maidul/sample.png
```

You now have: `wsnyolo.onnx yolov7tiny.onnx yolov5n.onnx yolov8n.onnx sample.png`.
Download them to your laptop (or scp straight to the Pi in Stage 3).

---

## Stage 2 — set up the Pi

1. Flash Raspberry Pi OS to the microSD (Raspberry Pi Imager), boot, connect to WiFi.
   Access it headless over SSH, or with a monitor + keyboard.
2. Enable I2C: `sudo raspi-config` -> Interface Options -> I2C -> Enable -> reboot.
3. Install the runtime:
   ```
   sudo apt update && sudo apt install -y python3-pip i2c-tools libopenblas-dev
   pip3 install onnxruntime numpy pillow pi-ina219
   ```
4. (Optional) fit a heatsink/fan on a Pi 4 so it doesn't thermally throttle.

## Stage 2b — wire the INA219 (for energy)

I2C side (4 jumper wires, Pi GPIO -> INA219):
- Pi 3.3V -> INA219 VCC
- Pi GND  -> INA219 GND
- Pi SDA (GPIO2, pin 3) -> INA219 SDA
- Pi SCL (GPIO3, pin 5) -> INA219 SCL

Power side (the cut USB cable): cut only the +5V wire of the Pi's USB power cable
and route it through the sensor — supply +5V -> INA219 Vin+, INA219 Vin- -> Pi +5V.
Leave GND and data wires intact.

Check the Pi sees the sensor: `i2cdetect -y 1` should show a device at `0x40`.

---

## Stage 3 — copy the kit + models to the Pi

```
mkdir -p ~/wp4/models
# copy measure_pi.py into ~/wp4/ , the four .onnx into ~/wp4/models/ , sample.png into ~/wp4/
```
(Use `scp` from the laptop, or a USB stick.)

---

## Stage 4 — run the measurement

Smoke test first, without energy (confirms the models run and gives latency/RAM/size):
```
cd ~/wp4
python3 measure_pi.py --models models/*.onnx --img sample.png --iters 50
```

Full run with energy (INA219 wired):
```
python3 measure_pi.py --models models/*.onnx --img sample.png --iters 100 --ina219
```

It prints one row per model: size_MB, mean/med/p95 latency (ms), RAM_MB, power_W,
energy_mJ. Paste that table back — it becomes the WP4 deployment table in the paper,
and WSN-YOLO should show the lowest latency and energy, strongest on the Pi Zero.

Repeat on both devices if you have them (Pi 4 = gateway, Pi Zero 2 W = node).
