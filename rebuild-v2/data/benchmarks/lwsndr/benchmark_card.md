# LWSNDR Transfer Benchmark v1.0 (built 2026-07-14, seed 0)

Source: Labelled WSN Data Repository (Suthaharan et al. 2010), 8 streams
({single,multi}-hop x {indoor,outdoor} x mote), humidity + temperature, 5s interval,
37,674 usable readings, 307 anomalous samples (0.81%) from physically introduced events.
Generator: src/injection/build_lwsndr.py. Downloaded from home.uncg.edu/cmp/downloads/lwsndr.html.

## Construction
- Windows: 64 samples x 2 modalities, OVERLAPPING stride 16 (events too sparse for
  non-overlapping: only 10 anomalous windows at stride 64).
- Leakage control: contiguous 128-sample blocks per stream; windows never cross block
  boundaries; whole blocks assigned to one split; anomalous blocks stratified separately
  with >= 1 anomalous block guaranteed in val and in test (seeded rng, frozen).
- Window label = 1 if any sample in window is labelled 1.
- Normalization: per modality, 1-99 pctl over all 8 streams, min-max to 0-255.
- Encoding: 256x256; humidity rows [0,128), temperature rows [128,256); time step = 4 cols.
- Labels: continuity convention, full-frame box, class 0 normal / 1 anomaly.

## Contents & stats
1,450 windows (23 anomalous); splits train/val/test = 1160/145/145 windows with
20/1/2 anomalous (blocks 232/29/29, anomalous blocks 4/1/1).
manifest.csv: name, stream, t_start, block, label.

## Intended use (per Master Plan)
Transfer/robustness check only, NOT a headline benchmark: tiny scale and extreme
imbalance are inherent to LWSNDR. Report accuracy, F1, AUROC and false alarms per
node per day; interpret test-split anomaly results qualitatively (n=2 windows).
