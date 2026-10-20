# WP3 Baseline Block: How To Run (DICC)

Code: src/baselines/ (features, rule, models, run_baseline, run_image_clf,
run_yolov8, table). Metrics: src/common/metrics.py (accuracy, F1, AUROC,
false alarms per node per day at 95% detection; threshold swept on val,
evaluated once on test). All raw/feature baselines run per (model, task, seed):

    # 7 raw/feature baselines x 2 tasks x 5 seeds (rule needs seed 0 only)
    for m in lstm_ae dense_ae cnn1d gru_clf ocsvm iforest; do
      for t in continuity lwsndr; do sbatch slurm/wp3_baselines.sbatch $m $t; done
    done
    python -m src.baselines.run_baseline --model rule --task continuity --seed 0
    python -m src.baselines.run_baseline --model rule --task lwsndr --seed 0

    # image classifiers on loc_hard binary (what classification cannot localise)
    for s in 0 1 2 3 4; do
      python -m src.baselines.run_image_clf --model mobilenetv3s --task loc_hard --seed $s
      python -m src.baselines.run_image_clf --model shufflenet  --task loc_hard --seed $s
    done

    # R4: off-the-shelf yolov8n (pip install ultralytics)
    python -m src.baselines.run_yolov8 --task loc_hard --seed 0     # + seeds 1-4
    python -m src.baselines.run_yolov8 --task continuity --data <continuity dataset.yaml> --seed 0

Merge parts + reproduce the table from the db alone:
    python -c "from src.common.db import merge_parts; print(merge_parts('results/results.sqlite','results/parts/*.sqlite'))"
    python -m src.baselines.table

Requirements on DICC: torch, torchvision, scikit-learn, ultralytics.
Continuity raw windows are cached on first run (data/benchmarks/wp3_cache/).
