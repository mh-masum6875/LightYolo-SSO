#!/bin/bash
#SBATCH --job-name=wp3
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-39%2
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
# ---------------------------------------------------------------------------
# WP3 cross-family baselines (reviewer R2.4 + R4), one submit runs everything.
#   sbatch slurm/wp3_run.sbatch
# Array index -> the Nth line of slurm/wp3_jobs.txt: "<driver> <model> <task> <seed>"
#   base   -> src.baselines.run_baseline   (raw-stream: AEs, CNN, GRU, ocsvm, iforest, rule) on faultstream
#   imgclf -> src.baselines.run_image_clf  (mobilenetv3s, shufflenet) on loc_hard binary
#   yolo   -> src.baselines.run_yolov8     (off-the-shelf yolov8n) on loc_hard detection
# %2 caps concurrency at the 2 idle GPUs. Submit from the yolov7_maidul root
# (same place you launched the gate and WP2 runs).
# ---------------------------------------------------------------------------
set -e
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate wp3
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs rebuild-v2/results/parts
export PYTHONPATH="$PWD/rebuild-v2:$PWD:$PYTHONPATH"
cd rebuild-v2

JOB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/wp3_jobs.txt)
read -r DRIVER MODEL TASK SEED <<< "$JOB"
echo "[$(date '+%F %T')] task $SLURM_ARRAY_TASK_ID -> driver=$DRIVER model=$MODEL task=$TASK seed=$SEED"
DB="results/parts/wp3_${DRIVER}_${MODEL}_${TASK}_s${SEED}.sqlite"

case "$DRIVER" in
  base)   python -m src.baselines.run_baseline  --model "$MODEL" --task "$TASK" --seed "$SEED" --db "$DB" ;;
  imgclf) python -m src.baselines.run_image_clf --model "$MODEL" --task "$TASK" --seed "$SEED" --db "$DB" ;;
  yolo)   python -m src.baselines.run_yolov8                     --task "$TASK" --seed "$SEED" --db "$DB" ;;
  *)      echo "unknown driver: $DRIVER" >&2; exit 2 ;;
esac
echo "[$(date '+%F %T')] task $SLURM_ARRAY_TASK_ID done -> $DB"
