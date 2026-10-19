#!/bin/bash
# WP4 one-shot Pi setup + base (no-sensor) measurement.
# Run from your home folder, where wp4_models.tgz and measure_pi.py also sit:
#     bash setup.sh
set -e

echo "=== [1/6] enabling I2C ==="
sudo raspi-config nonint do_i2c 0
sudo modprobe i2c-dev || true

echo "=== [2/6] installing system packages (this takes a minute) ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv i2c-tools libopenblas-dev

echo "=== [3/6] creating python environment + libraries ==="
python3 -m venv ~/wp4env
source ~/wp4env/bin/activate
pip install onnxruntime numpy pillow pi-ina219

echo "=== [4/6] unpacking models + script ==="
mkdir -p ~/wp4/models
cp measure_pi.py ~/wp4/
tar xzf wp4_models.tgz -C ~/wp4/models/
cd ~/wp4
python -c "from PIL import Image; Image.new('RGB',(640,640),(128,128,128)).save('sample.png')"

echo "=== [5/6] device info (for the paper) ==="
tr -d '\0' </proc/device-tree/model; echo
free -h | awk '/Mem/{print "RAM:",$2}'
grep PRETTY_NAME /etc/os-release
uname -r

echo "=== [6/6] sensor check — look for 40 in this grid ==="
i2cdetect -y 1

echo ""
echo "=== base measurement: size / latency / RAM (all 4 models) ==="
for m in models/wsnyolo.onnx models/yolov5n.onnx models/yolov8n.onnx models/yolov7tiny.onnx; do
  echo "### $m"
  python measure_pi.py --models "$m" --img sample.png --iters 100
done

echo ""
echo "=== DONE. If i2cdetect showed 40 and the tables printed, setup worked. ==="
echo "Next: power the Pi THROUGH the INA219, then run:  bash ~/measure_energy.sh"
