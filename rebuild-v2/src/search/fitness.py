"""WP2 shared fitness wrapper (Master Plan 6.1).

J(theta), MINIMIZED, identical for every strategy - the legacy multi-objective
form with the heuristic accuracy replaced by REAL proxy-trained mAP@0.5:

    J = 0.4*(1 - proxy_mAP50)
      + 0.3*(flops_g  / TARGET_FLOPS_G)
      + 0.2*(params_m / TARGET_PARAMS_M)
      + 0.1*(latency_ms / LATENCY_NORM_MS)
      + 10 * (flops_overrun + params_overrun)          # constraint penalties

Weights [0.4, 0.3, 0.2, 0.1] and targets (1.0 GFLOPs, 2.0 M params) are the
legacy Stage 1 values (WP0 audit). Proxy protocol (Master Plan 6.2): 20 epochs
on the fixed 25 percent stratified train subset, scored on the loc_hard val
split. Every evaluation logs one row per component to the parts database.

Backends:
  MockBackend  - deterministic analytic stand-in; local tests and dry runs.
  YoloBackend  - DICC: generates the arch yaml from theta, trains yolov7
                 train.py for --proxy-epochs on the proxy subset, parses
                 best mAP@0.5 from results.txt, measures params/GFLOPs.
                 Per-eval stdout/stderr go to <run_dir>/train.log so failures
                 are diagnosable. Systematic failure aborts fast (see
                 FitnessLogger.FAIL_FAST_AFTER).
"""
from __future__ import annotations
import os, subprocess, time
import numpy as np
from . import space

W_ACC, W_FLOPS, W_PARAMS, W_LAT = 0.4, 0.3, 0.2, 0.1
TARGET_FLOPS_G = 1.0
TARGET_PARAMS_M = 2.0
LATENCY_NORM_MS = 100.0

def compose_J(proxy_map50: float, flops_g: float, params_m: float, latency_ms: float) -> float:
    flops_pen = max(0.0, (flops_g - TARGET_FLOPS_G) / TARGET_FLOPS_G)
    params_pen = max(0.0, (params_m - TARGET_PARAMS_M) / TARGET_PARAMS_M)
    return (W_ACC * (1.0 - proxy_map50)
            + W_FLOPS * (flops_g / TARGET_FLOPS_G)
            + W_PARAMS * (params_m / TARGET_PARAMS_M)
            + W_LAT * (latency_ms / LATENCY_NORM_MS)
            + 10.0 * (flops_pen + params_pen))

class MockBackend:
    """Deterministic, cheap, and rugged enough to rank optimizers on:
    a shifted multimodal function mapped into plausible component ranges."""
    def __init__(self, **_):
        pass

    def evaluate(self, theta: np.ndarray) -> dict:
        t = space.discretize(theta)
        span = space.HIGH - space.LOW
        z = np.divide(t - space.LOW, span, out=np.zeros_like(t), where=span > 0)  # in [0,1]^15
        base = float(np.mean((z - 0.37) ** 2))              # optimum off-center
        ripple = 0.05 * float(np.mean(np.cos(6 * np.pi * z)))
        proxy_map50 = max(0.0, min(1.0, 0.9 - base - ripple))
        params_m = 0.5 + 3.5 * float(z[0] * 0.4 + z[5] * 0.3 + np.mean(z[1:4]) * 0.3)
        flops_g = 0.3 + 3.7 * float(np.mean(z[[0, 1, 2, 3, 5]]))
        latency_ms = 5.0 + 45.0 * flops_g / 4.0
        return dict(proxy_map50=proxy_map50, params_m=params_m,
                    flops_g=flops_g, latency_ms=latency_ms, wall_s=0.0, error="")

class YoloBackend:
    """Runs on DICC inside the yolov7 conda env. One evaluation = one proxy
    training (from scratch: --weights ''). A failed candidate returns
    proxy_map50=0 with the failure reason in `error`; the search continues
    and the failure is logged. train.log inside each eval dir has the full
    YOLOv7 output for diagnosis."""
    def __init__(self, yolov7_dir: str, data_yaml: str, hyp: str,
                 proxy_epochs: int = 20, img: int = 640, batch: int = 16,
                 device: str = "0", workdir: str = "runs/wp2",
                 weights: str = "yolov7-tiny.pt"):
        self.yolov7_dir = os.path.abspath(yolov7_dir)
        self.data_yaml = os.path.abspath(data_yaml)
        self.hyp = os.path.abspath(hyp)
        self.proxy_epochs, self.img, self.batch = proxy_epochs, img, batch
        self.device, self.workdir = device, workdir
        # Transfer from yolov7-tiny rather than training from scratch. The WP1 gate
        # reached 0.9183 with transfer and 150 epochs; from scratch at 20 epochs most
        # candidates never left the background-only regime (mAP ~0.005, flat).
        # yolov7 loads only the weights whose shapes match, so candidates that differ
        # from yolov7-tiny simply inherit fewer layers.
        self.weights = weights
        self._counter = 0

    def evaluate(self, theta: np.ndarray) -> dict:
        from .archgen import theta_to_cfg, measure_model
        t0 = time.time()
        self._counter += 1
        name = f"eval_{os.getpid()}_{self._counter:04d}"
        run_dir = os.path.join(self.yolov7_dir, self.workdir, name)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, "arch.yaml")
        err = ""
        theta_d = space.discretize(theta)
        try:
            theta_to_cfg(theta_d, cfg_path)
        except Exception as e:
            return dict(proxy_map50=0.0, params_m=99.0, flops_g=99.0,
                        latency_ms=9999.0, wall_s=time.time() - t0,
                        error=f"theta_to_cfg: {e!r}")
        try:
            params_m, flops_g, latency_ms = measure_model(cfg_path, img=self.img)
        except Exception as e:
            # unbuildable candidate: do not waste GPU time trying to train it
            return dict(proxy_map50=0.0, params_m=99.0, flops_g=99.0,
                        latency_ms=9999.0, wall_s=time.time() - t0,
                        error=f"measure_model (unbuildable, training skipped): {e!r}")
        proxy_map50 = 0.0
        cmd = ["python", "train.py", "--weights", self.weights,
               "--cfg", cfg_path, "--data", self.data_yaml,
               "--hyp", self.hyp, "--epochs", str(self.proxy_epochs),
               "--img-size", str(self.img), str(self.img),
               "--batch-size", str(self.batch), "--device", self.device,
               "--project", os.path.join(self.workdir, name), "--name", "proxy",
               "--exist-ok", "--noautoanchor", "--nosave"]
        log_path = os.path.join(run_dir, "train.log")
        try:
            with open(log_path, "w") as lf:
                subprocess.run(cmd, cwd=self.yolov7_dir, check=True,
                               stdout=lf, stderr=subprocess.STDOUT,
                               timeout=3600 * 6)
            results = os.path.join(self.yolov7_dir, self.workdir, name, "proxy", "results.txt")
            proxy_map50 = _best_map50(results)
        except Exception as e:
            err = (err + "; " if err else "") + f"train: {e!r} (see {log_path})"
        return dict(proxy_map50=proxy_map50, params_m=params_m, flops_g=flops_g,
                    latency_ms=latency_ms, wall_s=time.time() - t0, error=err)

class CostBackend:
    """CPU-cheap efficiency backend (R2.3 optimizer contest under the
    efficiency-frontier framing).

    Builds the real architecture from theta and measures its TRUE params/GFLOPs
    with thop (no training). Accuracy is not proxy-trained (the 20-epoch proxy
    was dead, Spearman rho=0.048, and the WP1 frontier showed mAP@0.5 is FLAT,
    0.905-0.930, across the whole feasible space). Instead accuracy is a
    surrogate calibrated to the 9 measured frontier points -- a mild log-capacity
    rise that saturates -- so J keeps the paper's exact multi-objective form and
    has an interior optimum at the efficiency knee. Every optimizer minimises the
    identical J; the winner is a REAL, buildable architecture whose accuracy band
    is confirmed by a single full training afterwards.

    Frontier calibration (WSN-FaultBench loc_hard, 640px):
      (params_m, mAP@0.5): (1.17, 0.905) ... (6.02, 0.930)  -> acc ~= A + B*ln(params_m)
    """
    A_ACC, B_ACC = 0.9026, 0.01526     # least-squares on the frontier endpoints
    ACC_CAP = 0.930                    # measured frontier ceiling
    ACC_FLOOR_CLIP = 0.60              # keeps unbuildable/degenerate archs low

    def __init__(self, yolov7_dir: str, base_cfg: str | None = None,
                 img: int = 640, latency_iters: int = 20, **_):
        self.yolov7_dir = os.path.abspath(yolov7_dir)
        self.base_cfg = base_cfg
        self.img, self.latency_iters = img, latency_iters
        self._counter = 0

    def _surrogate_acc(self, params_m: float) -> float:
        import math
        if params_m <= 0:
            return 0.0
        acc = self.A_ACC + self.B_ACC * math.log(params_m)
        return max(0.0, min(self.ACC_CAP, acc))

    LAT_MS_PER_GFLOP = 2.0            # CPU-agnostic latency proxy (monotone in FLOPs)

    def _measure(self, cfg_path: str):
        """Real params (M) + thop GFLOPs on CPU, single forward pass, NO timing
        loop (that loop is the CPU bottleneck and latency here just scales with
        FLOPs anyway)."""
        import copy as _copy
        import torch
        from models.yolo import Model
        model = Model(cfg_path, ch=3, nc=4).eval()      # CPU
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.zeros(1, 3, self.img, self.img)
        from thop import profile
        flops, _ = profile(_copy.deepcopy(model), inputs=(x,), verbose=False)
        flops_g = flops * 2 / 1e9
        del model, x
        return float(params_m), float(flops_g)

    def evaluate(self, theta: np.ndarray) -> dict:
        from .archgen import theta_to_cfg
        t0 = time.time()
        self._counter += 1
        run_dir = os.path.join(self.yolov7_dir, "runs", "wp2_cost")
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, f"arch_{os.getpid()}_{self._counter:05d}.yaml")
        theta_d = space.discretize(theta)
        try:
            theta_to_cfg(theta_d, cfg_path, base_cfg=self.base_cfg)
            params_m, flops_g = self._measure(cfg_path)
        except Exception as e:
            # unbuildable candidate: heavy penalty so the search avoids it
            return dict(proxy_map50=0.0, params_m=99.0, flops_g=99.0,
                        latency_ms=9999.0, wall_s=time.time() - t0,
                        error=f"cost-eval (unbuildable): {e!r}")
        finally:
            try:
                os.remove(cfg_path)
            except OSError:
                pass
        latency_ms = flops_g * self.LAT_MS_PER_GFLOP
        acc = self._surrogate_acc(params_m)
        return dict(proxy_map50=acc, params_m=params_m, flops_g=flops_g,
                    latency_ms=latency_ms, wall_s=time.time() - t0, error="")

def _best_map50(results_txt: str) -> float:
    best = 0.0
    with open(results_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 11:
                try:
                    best = max(best, float(parts[10]))
                except ValueError:
                    continue
    return best

BACKENDS = {"mock": MockBackend, "yolo": YoloBackend, "cost": CostBackend}

class FitnessLogger:
    """Wraps a backend; computes J; writes one row per component per
    evaluation into the (parts) database. run_id = wp2_{strategy}_s{seed}_e{idx:03d}.

    FAIL-FAST: if the first FAIL_FAST_AFTER evaluations ALL fail
    (proxy_map50 == 0 with an error), something systematic is broken
    (environment, paths, cfg incompatibility) - abort loudly instead of
    burning the budget on failures."""
    FAIL_FAST_AFTER = 3

    def __init__(self, backend, db_path: str, strategy: str, seed: int, dataset: str = "loc_hard"):
        self.backend, self.db_path = backend, db_path
        self.strategy, self.seed, self.dataset = strategy, seed, dataset
        self.n_evals = 0
        self.n_failures = 0
        self.history: list[tuple[int, float]] = []      # (eval_idx, J)

    def __call__(self, theta: np.ndarray) -> float:
        from src.common.db import log_metric
        comp = self.backend.evaluate(theta)
        if comp.get("error"):
            self.n_failures += 1
            print(f"[eval {self.n_evals}] FAILED: {comp['error']}", flush=True)
        if (self.n_evals + 1 == self.FAIL_FAST_AFTER
                and self.n_failures == self.FAIL_FAST_AFTER):
            raise RuntimeError(
                f"first {self.FAIL_FAST_AFTER} evaluations all failed - "
                f"systematic problem, aborting. Last error: {comp['error']}")
        J = compose_J(comp["proxy_map50"], comp["flops_g"], comp["params_m"], comp["latency_ms"])
        idx = self.n_evals
        run_id = f"wp2_{self.strategy}_s{self.seed}_e{idx:03d}"
        tjson = space.theta_to_json(space.discretize(theta))
        for metric, value in [("J", J), ("proxy_map50", comp["proxy_map50"]),
                              ("params_m", comp["params_m"]), ("flops_g", comp["flops_g"]),
                              ("latency_ms", comp["latency_ms"]), ("wall_s", comp["wall_s"])]:
            log_metric(self.db_path, run_id, metric, float(value), wp="WP2",
                       task="search", dataset=self.dataset, tier="hard",
                       method=self.strategy, model="theta", seed=self.seed,
                       split="val", config_path=tjson,
                       gpu_hours=comp["wall_s"] / 3600.0 if metric == "wall_s" else None)
        self.n_evals += 1
        self.history.append((idx, J))
        return J
