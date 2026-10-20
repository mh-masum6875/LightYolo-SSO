#!/bin/bash
#SBATCH --job-name=loclws
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
# Second-dataset detector runs on loc_lwsndr (localized faults on independent
# LWSNDR signal). Trains WSN-YOLO (arch_07 cfg) and YOLOv7-tiny on the SAME
# benchmark. Submit from the yolov7_maidul root:
#   sbatch rebuild-v2/slurm/loc_lwsndr_yolov7.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate yolov7_maidul
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CFGS=(frontier_archs/arch_07.yaml cfg/training/yolov7-tiny.yaml)
NAMES=(wsnyolo yolov7tiny)
i=$SLURM_ARRAY_TASK_ID
CFG=${CFGS[$i]}; NAME=${NAMES[$i]}
echo "[$(date '+%F %T')] loc_lwsndr training: $NAME  cfg=$CFG"

python train.py --weights yolov7-tiny.pt --cfg "$CFG" \
  --data rebuild-v2/data/benchmarks/loc_lwsndr/dataset_abs.yaml \
  --img-size 640 640 --batch-size 16 --epochs 100 \
  --name "loc_lwsndr_${NAME}" --project runs/loc_lwsndr --exist-ok --device 0 --noautoanchor

# report best mAP@0.5 and mAP@0.5:0.95
R="runs/loc_lwsndr/loc_lwsndr_${NAME}/results.txt"
awk 'NF>=12 {if($11>m){m=$11; m95=$12}} END{printf "['"$NAME"'] best mAP@0.5=%.4f  mAP@0.5:0.95=%.4f\n", m, m95}' "$R"
echo "[$(date '+%F %T')] done $NAME"
