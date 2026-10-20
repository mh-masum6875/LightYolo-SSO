#!/bin/bash
#SBATCH --job-name=wp3v5
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2%2
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
# ---------------------------------------------------------------------------
# Classic YOLOv5n detector baseline on WSN-FaultBench loc_hard (3 seeds).
# Uses the Dependencies/yolov5 repo in the yolov5_custom env, mirroring how v5n
# was trained in the rejected paper. Logs mAP to results/parts so wp3_collect
# folds it into the same table as yolov8n.
#   Submit from the yolov7_maidul root:  sbatch rebuild-v2/slurm/wp3_yolov5.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate yolov5_custom
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs rebuild-v2/results/parts
export PYTHONPATH="$PWD/rebuild-v2:$PYTHONPATH"
SEED=$SLURM_ARRAY_TASK_ID
echo "[$(date '+%F %T')] yolov5n loc_hard seed=$SEED"
python rebuild-v2/src/baselines/run_yolov5.py \
  --seed "$SEED" \
  --yolov5-dir "$HOME/Dependencies/yolov5" \
  --data "$PWD/rebuild-v2/data/benchmarks/loc_hard/dataset_yolov5.yaml" \
  --db   "$PWD/rebuild-v2/results/parts/wp3_yolov5n_loc_hard_s${SEED}.sqlite"
echo "[$(date '+%F %T')] yolov5n seed=$SEED done"
