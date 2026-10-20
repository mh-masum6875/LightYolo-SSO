#!/bin/bash
#SBATCH --job-name=wp3v5val
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2%2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# ---------------------------------------------------------------------------
# Evaluate the ALREADY-TRAINED yolov5n best.pt (seeds 0-2) on the loc_hard TEST
# split and log mAP + params + FLOPs. No retraining. Fixes the PyTorch>=2.6
# post-training crash via the compat shim. Submit from the yolov7_maidul root:
#   sbatch rebuild-v2/slurm/wp3_yolov5_val.sbatch
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
echo "[$(date '+%F %T')] yolov5n VAL-ONLY loc_hard seed=$SEED"
python rebuild-v2/src/baselines/run_yolov5.py \
  --val-only --seed "$SEED" \
  --yolov5-dir "$HOME/Dependencies/yolov5" \
  --data "$PWD/rebuild-v2/data/benchmarks/loc_hard/dataset_yolov5.yaml" \
  --db   "$PWD/rebuild-v2/results/parts/wp3_yolov5n_loc_hard_s${SEED}.sqlite"
echo "[$(date '+%F %T')] seed=$SEED done"
