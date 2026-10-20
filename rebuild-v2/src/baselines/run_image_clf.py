"""WP3 image classifiers: mobilenetv3s / shufflenetv2_x0.5 (torchvision,
1-channel input, supervised) on continuity, lwsndr, and loc_hard binary
(to show what classification cannot localise). 5 seeds.

Usage (DICC):
  python -m src.baselines.run_image_clf --model mobilenetv3s --task loc_hard --seed 0
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.common.metrics import evaluate_scores
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def build(model_name: str):
    import torch.nn as nn
    from torchvision import models as tvm
    # num_classes=2 alone builds an un-pretrained model on every torchvision
    # version; the newer weights=None kwarg is rejected by older torchvision
    # (which shufflenet_v2_x0_5 uses here), so we avoid it.
    if model_name == "mobilenetv3s":
        m = tvm.mobilenet_v3_small(num_classes=2)
        m.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
    elif model_name == "shufflenet":
        m = tvm.shufflenet_v2_x0_5(num_classes=2)
        m.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False)
    else:
        raise ValueError(model_name)
    return m

class PngSet:
    def __init__(self, bench, ids, labels):
        self.bench, self.ids, self.labels = bench, ids, labels
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        import torch
        from PIL import Image
        x = np.array(Image.open(os.path.join(self.bench, "images", self.ids[i] + ".png")),
                     dtype=np.float32)[None] / 255.0
        return torch.tensor(x), int(self.labels[self.ids[i]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("mobilenetv3s", "shufflenet"), required=True)
    ap.add_argument("--task", choices=("loc_hard", "continuity", "lwsndr"), default="loc_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--db", default=os.path.join(ROOT, "results", "results.sqlite"))
    a = ap.parse_args()
    import torch, torch.nn as nn
    from src.baselines import data as D
    from src.baselines.models import _seed_all
    _seed_all(a.seed)
    if a.task == "loc_hard":
        bench, splits, labels = D.load_loc_hard_binary()
    else:
        raise SystemExit("continuity/lwsndr image runs use the curated image folders; see WP3_RUN_README")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build(a.model).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    dl = torch.utils.data.DataLoader(PngSet(bench, splits["train"], labels),
                                     batch_size=a.batch, shuffle=True, num_workers=4)
    net.train()
    for _ in range(a.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    def scores(ids):
        net.eval(); out = []
        dl = torch.utils.data.DataLoader(PngSet(bench, ids, labels), batch_size=a.batch, num_workers=4)
        with torch.no_grad():
            for xb, _ in dl:
                out.append(torch.softmax(net(xb.to(dev)), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)
    y = lambda ids: np.array([labels[i] for i in ids])
    res = evaluate_scores(y(splits["val"]), scores(splits["val"]),
                          y(splits["test"]), scores(splits["test"]), n_nodes=55)
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    run_id = f"wp3_{a.model}_{a.task}_s{a.seed}"
    from src.common.db import log_metric
    for k in ("accuracy", "f1", "auroc", "false_alarms_per_node_day", "recall_test"):
        log_metric(a.db, run_id, k, float(res[k]), wp="WP3", task=a.task, dataset=a.task,
                   method=a.model, model=a.model, seed=a.seed, split="test",
                   gpu_hours=(time.time() - t0) / 3600.0 if k == "accuracy" else None)
    log_metric(a.db, run_id, "params_m", params_m, wp="WP3", task=a.task, dataset=a.task,
               method=a.model, model=a.model, seed=a.seed, split="test")
    print(run_id, {k: round(float(v), 4) for k, v in res.items()})

if __name__ == "__main__":
    main()
