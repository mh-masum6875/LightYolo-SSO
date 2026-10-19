#!/bin/bash
# WP4 energy measurement. Run ONLY after the Pi is powered through the INA219
# (sacrificial USB-A->C cable: red +5V wire cut and routed through Vin+/Vin-).
#     bash ~/measure_energy.sh
set -e
cd ~/wp4
source ~/wp4env/bin/activate

echo "=== confirming sensor is visible (look for 40) ==="
i2cdetect -y 1

echo ""
echo "=== ENERGY measurement (size / latency / RAM / power / energy) ==="
for m in models/wsnyolo.onnx models/yolov5n.onnx models/yolov8n.onnx models/yolov7tiny.onnx; do
  echo "### $m"
  python measure_pi.py --models "$m" --img sample.png --iters 100 --ina219
done

echo ""
echo "=== energy measurement DONE. Screenshot this, then run:  sudo poweroff ==="
