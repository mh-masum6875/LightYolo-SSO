#!/bin/bash
#SBATCH --job-name=wp3
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch slurm/wp3_baselines.sbatch <model> <task>
#   model: lstm_ae dense_ae cnn1d gru_clf ocsvm iforest rule
#   task:  continuity lwsndr
# Array index = seed (rule/ocsvm/iforest are deterministic-ish; still 5 seeds for uniformity).
set -e
MODEL=${1:?model}; TASK=${2:?task}
module load miniconda
conda activate yolov7_maidul
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$PWD/rebuild-v2:$PYTHONPATH"
cd rebuild-v2
python -m src.baselines.run_baseline --model "$MODEL" --task "$TASK" --seed "$SLURM_ARRAY_TASK_ID" \
  --db "results/parts/wp3_${MODEL}_${TASK}_s${SLURM_ARRAY_TASK_ID}.sqlite"
