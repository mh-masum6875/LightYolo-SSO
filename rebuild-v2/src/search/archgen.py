"""theta -> yolov7 architecture yaml. Port of the legacy
YOLOArchitectureGenerator with its latent bugs FIXED (the legacy Stage 1
never built its candidates - heuristic fitness - so these were never hit):

BUGFIX 1 (legacy): channel scaling applied to args[0] of EVERY layer,
  including Concat's dimension and SP's kernel size. Now only Conv-family
  layers are scaled.
BUGFIX 2 (legacy): Conv layers in yolov7-tiny carry 6 args
  [c2, k, s, p, g, act]; substituting the module name to DWConv/GhostConv
  keeps those args, which do not fit their signatures. Substituted layers
  now get exactly [c2, k, s].

Capability guard: DWConv/GhostConv substitutions apply only if the classes
exist in models/common.py of the local repo (checked once, warning printed).
measure_model() gives real params (M), thop GFLOPs, and median GPU latency.
torch/yaml imports are function-local so the module imports in the sandbox.
"""
from __future__ import annotations
import copy, os
import numpy as np
from . import space

BASE_CFG = os.environ.get("WP2_BASE_CFG", "cfg/training/yolov7-tiny.yaml")
NC = 4  # WSN-FaultBench classes: spike, stuck, drift, noise
SCALABLE = {"Conv", "DWConv", "GhostConv", "RepConv", "SPPCSPC"}  # channel arg 0
_CAPS: dict | None = None

def _caps() -> dict:
    global _CAPS
    if _CAPS is None:
        caps = {}
        try:
            import models.common as C
            caps["DWConv"] = hasattr(C, "DWConv")
            caps["GhostConv"] = hasattr(C, "GhostConv")
        except Exception:
            caps = {"DWConv": True, "GhostConv": True}   # off-DICC: assume ok
        for k, v in caps.items():
            if not v:
                print(f"WARNING archgen: {k} not in models/common.py - "
                      f"that substitution is a NO-OP in this repo", flush=True)
        _CAPS = caps
    return _CAPS

def theta_to_cfg(theta: np.ndarray, out_path: str, base_cfg: str | None = None) -> dict:
    import yaml
    t = space.discretize(theta)
    (initial_channels, w_p3, w_p4, w_p5, head_depth_mult, head_base_channels,
     _conf, _nms, a_p3, a_p4, a_p5, csp_bb, csp_head, use_sep, use_ghost) = t
    caps = _caps()
    use_sep = bool(use_sep > 0.5) and caps["DWConv"]
    use_ghost = bool(use_ghost > 0.5) and caps["GhostConv"]
    with open(base_cfg or BASE_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = NC
    cfg["backbone"] = _scale_layers(cfg["backbone"], stage_scales(w_p3, w_p4, w_p5),
                                    use_sep, use_ghost)
    cfg["head"] = _scale_layers(cfg["head"], None, use_sep, use_ghost,
                                uniform_scale=float(head_depth_mult))
    cfg["anchors"] = _scale_anchors(cfg["anchors"], [float(a_p3), float(a_p4), float(a_p5)])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    return cfg

def stage_scales(w_p3: float, w_p4: float, w_p5: float):
    """Legacy stage boundaries: layers <8 P1-P2 (uses w_p3), <15 P3, <22 P4, rest P5."""
    def scale_for(i: int) -> float:
        if i < 8: return w_p3
        if i < 15: return w_p3
        if i < 22: return w_p4
        return w_p5
    return scale_for

def _is_conv(layer) -> bool:
    return (len(layer) > 3 and isinstance(layer[2], str) and layer[2] in SCALABLE
            and isinstance(layer[3], list) and layer[3] and isinstance(layer[3][0], int))

def _activation_of(args):
    """Return the activation argument of a yolov7 Conv layer, or None.
    yolov7-tiny Conv args are [c2, k, s, p, g, act]; act is a string such as
    "nn.LeakyReLU(0.1)" once loaded from yaml. Anything shorter has no explicit
    activation and therefore uses the class default."""
    if len(args) >= 6 and isinstance(args[5], str):
        return args[5]
    return None

def _scale_layers(layers, scale_for, use_sep: bool, use_ghost: bool,
                  uniform_scale: float | None = None):
    out = []
    for i, layer in enumerate(layers):
        new_layer = copy.deepcopy(layer)
        if _is_conv(layer):                          # BUGFIX 1: Conv-family only
            s = uniform_scale if uniform_scale is not None else scale_for(i)
            new_layer[3][0] = max(8, int(layer[3][0] * s))
            k = layer[3][1] if len(layer[3]) > 1 and isinstance(layer[3][1], int) else 1
            st = layer[3][2] if len(layer[3]) > 2 and isinstance(layer[3][2], int) else 1
            act = _activation_of(layer[3])           # BUGFIX 3: keep the activation
            if layer[2] == "Conv":
                # BUGFIX 3 (2026-07-22): the previous version replaced the argument
                # list with [c2, k, s], which silently dropped nn.LeakyReLU(0.1) and
                # let those layers fall back to the yolov7 default activation, SiLU.
                # In the seed-0 random search that made theta dimension 13 change the
                # activation function as well as the convolution type: 13 of 13 DWConv
                # candidates learned while 0 of 44 GhostConv and 1 of 16 plain Conv did.
                # The search was measuring SiLU against LeakyReLU, not architecture.
                # Substituted layers now carry the same activation as the layer they
                # replace, so only the convolution type varies.
                if use_ghost:                        # ghost wins if both flags set
                    new_layer[2] = "GhostConv"
                    # GhostConv(c1, c2, k, s, g, act): groups sits before act
                    new_layer[3] = ([new_layer[3][0], k, st, 1, act] if act
                                    else [new_layer[3][0], k, st])
                elif use_sep and k == 3:
                    new_layer[2] = "DWConv"
                    # DWConv(c1, c2, k, s, act): act is the fifth argument
                    new_layer[3] = ([new_layer[3][0], k, st, act] if act
                                    else [new_layer[3][0], k, st])
        out.append(new_layer)
    return out

def _scale_anchors(anchors, scales):
    out = []
    for i, anchor_set in enumerate(anchors):
        s = scales[min(i, len(scales) - 1)]
        scaled = []
        for j in range(0, len(anchor_set), 2):
            scaled += [max(1, int(anchor_set[j] * s)), max(1, int(anchor_set[j + 1] * s))]
        out.append(scaled)
    return out

def measure_model(cfg_path: str, img: int = 640, latency_iters: int = 50):
    """Real params (M), FLOPs (G, thop), and GPU forward latency (ms, median).
    Must run inside the yolov7 repo (models.yolo importable). ch=3 to match
    the training input pipeline."""
    import torch
    from models.yolo import Model                      # yolov7 repo import
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(cfg_path, ch=3, nc=NC).to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.zeros(1, 3, img, img, device=device)
    try:
        from thop import profile
        flops, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
        flops_g = flops * 2 / 1e9                      # MACs -> FLOPs convention
    except Exception:
        flops_g = 99.0
    times = []
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time as _t
        for _ in range(latency_iters):
            t0 = _t.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((_t.perf_counter() - t0) * 1000)
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(params_m), float(flops_g), float(np.median(times))
