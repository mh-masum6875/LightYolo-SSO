#!/bin/bash
#SBATCH --job-name=wp2
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-49%3
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# ---------------------------------------------------------------------------
# WP2 R2.3 optimizer contest, EFFICIENCY framing. Each job = one (strategy,seed)
# search of --budget evaluations on the CPU-cheap CostBackend (real params/FLOPs
# via thop, no training; accuracy from the frontier-calibrated surrogate). This
# is CPU work; the GPU is only requested to reuse the known-good partition config
# and sits idle. ~3s/eval, so a 100-eval run is ~5 min.
#   Submit from the yolov7_maidul root:  sbatch rebuild-v2/slurm/wp2_cost.sbatch
# Array index -> Nth line of slurm/wp2_jobs.txt: "<strategy> <seed>".
# ---------------------------------------------------------------------------
set -e
BUDGET=${1:-100}
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate yolov7_maidul
cd "$SLURM_SUBMIT_DIR"                 # = yolov7_maidul repo root (models/ here)
mkdir -p logs rebuild-v2/results/parts
python -m pip show optuna >/dev/null 2>&1 || pip install --user optuna
python -m pip show thop   >/dev/null 2>&1 || pip install --user thop
export PYTHONPATH="$PWD/rebuild-v2:$PWD:$PYTHONPATH"

JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" rebuild-v2/slurm/wp2_jobs.txt)
read -r STRATEGY SEED <<< "$JOB"
echo "[$(date '+%F %T')] WP2 cost: strategy=$STRATEGY seed=$SEED budget=$BUDGET"
python -m src.search.run \
  --strategy "$STRATEGY" --seed "$SEED" --budget "$BUDGET" \
  --backend cost --yolov7-dir "$PWD" \
  --base-cfg "$PWD/cfg/training/yolov7-tiny.yaml" --img 640 \
  --db "rebuild-v2/results/parts/wp2_${STRATEGY}_s${SEED}.sqlite"
echo "[$(date '+%F %T')] done strategy=$STRATEGY seed=$SEED"
