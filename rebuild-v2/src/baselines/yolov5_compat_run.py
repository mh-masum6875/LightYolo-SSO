"""Run a classic yolov5 script (train.py / val.py) under PyTorch >= 2.6.

PyTorch 2.6 changed torch.load's default to weights_only=True, which breaks the
classic ultralytics/yolov5 repo (its strip_optimizer and checkpoint loads call
torch.load without weights_only=False and hit UnpicklingError on numpy globals).
This shim restores weights_only=False, then runs the requested yolov5 script as
__main__ with its own argv.

Usage (cwd must be the yolov5 repo root):
    python yolov5_compat_run.py train.py --weights ... --data ...
    python yolov5_compat_run.py val.py   --weights ... --data ... --task test
"""
import sys
import runpy
import torch

_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _load

if len(sys.argv) < 2:
    raise SystemExit("usage: python yolov5_compat_run.py <script.py> [args...]")

script = sys.argv[1]
sys.argv = sys.argv[1:]          # the target script sees its own name + args
runpy.run_path(script, run_name="__main__")
