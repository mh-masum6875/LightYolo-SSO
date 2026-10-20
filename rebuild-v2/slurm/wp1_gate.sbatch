#!/bin/bash
#SBATCH --job-name=wp1_gate --gres=gpu:1
#SBATCH --time=24:00:00 --output=logs/%x.out
# WP1 gate run: existing WSN-YOLO configuration, once, on loc_hard (Master Plan 5.3).
# Pass: mAP@0.5 in [0.80, 0.95]. >0.97: tighten hard tier. <0.70: widen. Log every attempt.
# Before first use: dos2unix this file; check torch.load weights_only=False note (Plan 11).
source ~/venv/bin/activate
cd $SLURM_SUBMIT_DIR   # = LightYolo-SSO repo root with rebuild-v2/ alongside, see GATE_RUN_README
python train.py \
  --weights yolov7-tiny.pt \
  --data ../rebuild-v2/data/benchmarks/loc_hard/dataset.yaml \
  --cfg ../rebuild-v2/configs/yolo_wsn_faultbench_nc4.yaml \
  --hyp data/hyp.wsn.yaml \
  --img 640 640 --batch-size 16 --epochs 150 \
  --name wp1_gate_loc_hard --seed 0
