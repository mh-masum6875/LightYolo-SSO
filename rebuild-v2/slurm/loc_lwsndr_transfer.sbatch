#!/bin/bash
#SBATCH --job-name=loclwstf
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# ---------------------------------------------------------------------------
# Second-dataset TRANSFER evaluation on loc_lwsndr. Both detectors are fine-tuned
# from their loc_hard-trained weights (the correct protocol for a transfer/
# generalization dataset), not from generic COCO weights.
#   WSN-YOLO : cfg arch_07,  init runs/frontier/arch_07/weights/best.pt
#   YOLOv7-tiny(full-width arch_05): cfg arch_05, init runs/frontier/arch_05/weights/best.pt
# Submit from the yolov7_maidul root:  sbatch rebuild-v2/slurm/loc_lwsndr_transfer.sbatch
# Args verified against this repo's train.py usage (no --patience).
# ---------------------------------------------------------------------------
set -e
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate yolov7_maidul
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CFGS=(frontier_archs/arch_07.yaml frontier_archs/arch_05.yaml)
WEIGHTS=(runs/frontier/arch_07/weights/best.pt runs/frontier/arch_05/weights/best.pt)
NAMES=(wsnyolo_tf yolov7tiny_tf)
i=$SLURM_ARRAY_TASK_ID
CFG=${CFGS[$i]}; W=${WEIGHTS[$i]}; NAME=${NAMES[$i]}
echo "[$(date '+%F %T')] loc_lwsndr TRANSFER: $NAME  cfg=$CFG  init=$W"

python train.py --weights "$W" --cfg "$CFG" \
  --data rebuild-v2/data/benchmarks/loc_lwsndr/dataset_abs.yaml \
  --img-size 640 640 --batch-size 16 --epochs 100 \
  --name "loc_lwsndr_${NAME}" --project runs/loc_lwsndr --exist-ok --device 0 --noautoanchor

R="runs/loc_lwsndr/loc_lwsndr_${NAME}/results.txt"
awk 'NF>=12 {if($11>m){m=$11; m95=$12}} END{printf "['"$NAME"'] best mAP@0.5=%.4f  mAP@0.5:0.95=%.4f\n", m, m95}' "$R"
echo "[$(date '+%F %T')] done $NAME"
