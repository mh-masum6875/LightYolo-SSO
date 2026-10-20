#!/bin/bash -l
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=wp1_gate
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# WP1 GATE RUN - WSN-FaultBench difficulty check (Master Plan v2, section 5.3)
# Trains the EXISTING SSO architecture (only nc changed 2->4) once on loc_hard.
# PASS if best mAP@0.5 is between 0.80 and 0.95.

module purge
module load miniconda

conda activate yolov7_maidul || exit 1

cd /home/user/mhmasum6875/yolov7_maidul || exit 1

# if the uploaded folder has a space in its name, fix it once
[ -d "rebuild v2" ] && [ ! -d "rebuild-v2" ] && mv "rebuild v2" rebuild-v2

# sanity checks before burning GPU hours
for f in rebuild-v2/configs/yolo_wsn_faultbench_nc4.yaml \
         rebuild-v2/configs/loc_hard_dicc.yaml \
         rebuild-v2/data/benchmarks/loc_hard/train_paths.txt \
         data/hyp.wsn.yaml yolov7-tiny.pt train.py; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
N=$(ls rebuild-v2/data/benchmarks/loc_hard/images | wc -l)
echo "loc_hard images found: $N (expect 14682)"
[ "$N" -ge 14000 ] || { echo "loc_hard images incomplete - re-upload rebuild-v2"; exit 1; }

# avoid YOLOv7 auto-resume trap
rm -rf runs/train/wp1_gate_loc_hard*

echo "===== GPU STATUS ====="
nvidia-smi

python train.py \
  --weights yolov7-tiny.pt \
  --cfg rebuild-v2/configs/yolo_wsn_faultbench_nc4.yaml \
  --data rebuild-v2/configs/loc_hard_dicc.yaml \
  --hyp data/hyp.wsn.yaml \
  --img-size 640 640 \
  --batch-size 16 \
  --epochs 150 \
  --name wp1_gate_loc_hard

echo "===== GATE RESULT ====="
python - << 'PY'
best = 0.0
try:
    for line in open("runs/train/wp1_gate_loc_hard/results.txt"):
        p = line.split()
        if len(p) >= 11:
            try: best = max(best, float(p[10]))
            except ValueError: pass
except FileNotFoundError:
    print("results.txt not found - training failed"); raise SystemExit(1)
verdict = ("PASS - benchmark difficulty OK, WP2 can start" if 0.80 <= best <= 0.95
           else "TOO EASY - faults must be made subtler" if best > 0.97
           else "TOO HARD - faults must be made stronger" if best < 0.70
           else "BORDERLINE - discuss in next Cowork session")
print(f"BEST mAP@0.5 = {best:.4f}  ->  {verdict}")
PY
