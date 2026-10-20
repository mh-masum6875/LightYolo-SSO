#!/bin/bash
#SBATCH --job-name=wp2
#SBATCH --partition=gpu-titan
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch slurm/wp2_array.sbatch <strategy>
#   strategy in: sso pso random revol tpe
# Run from the yolov7 repo root on DICC (same layout as the gate run):
#   /home/user/mhmasum6875/yolov7_maidul/  with rebuild-v2/ inside it.
set -e
STRATEGY=${1:?usage: sbatch slurm/wp2_array.sbatch <strategy>}
module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate yolov7_maidul
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs rebuild-v2/results/parts
python -m pip show optuna >/dev/null 2>&1 || pip install --user optuna
export PYTHONPATH="$PWD/rebuild-v2:$PWD:$PYTHONPATH"
python -m src.search.run \
  --strategy "$STRATEGY" --seed "$SLURM_ARRAY_TASK_ID" --budget 50 \
  --backend yolo --yolov7-dir "$PWD" \
  --data rebuild-v2/configs/loc_hard_proxy_dicc.yaml \
  --hyp data/hyp.wsn.yaml --weights yolov7-tiny.pt --proxy-epochs 20 --img 640 --batch 16 --device 0 \
  --db "rebuild-v2/results/parts/wp2_${STRATEGY}_s${SLURM_ARRAY_TASK_ID}.sqlite"
