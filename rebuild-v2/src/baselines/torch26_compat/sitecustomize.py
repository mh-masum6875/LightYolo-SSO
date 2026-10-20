"""Auto-loaded by Python at interpreter startup when this directory is on
PYTHONPATH. Restores torch.load's pre-2.6 behaviour (weights_only=False) so the
classic ultralytics/yolov5 repo runs unmodified under PyTorch >= 2.6. Scripts
(train.py / val.py) are then launched directly, so their `models` package
imports resolve normally.
"""
try:
    import torch

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load
except Exception:
    # torch not importable at startup for some other tool: do nothing.
    pass
