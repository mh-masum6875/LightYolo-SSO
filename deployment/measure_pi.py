"""WP4 on-device deployment measurement (Raspberry Pi).

For each ONNX detector, runs onnxruntime inference and reports:
  - model size (MB, the .onnx file)
  - latency per image (ms): mean / median / p95
  - peak RAM (MB)
  - active power (W) and energy per inference (mJ), if an INA219 is wired (--ina219)

Energy is computed as (active_power - idle_power) x latency, i.e. the net energy
the inference itself draws on top of the Pi's idle baseline.

Usage (on the Pi):
  python3 measure_pi.py --models models/*.onnx --img sample.png --iters 100
  python3 measure_pi.py --models models/*.onnx --img sample.png --iters 100 --ina219
"""
import argparse, glob, os, time, threading
import numpy as np


def make_input(img_path, size=640):
    from PIL import Image
    im = Image.open(img_path).convert("RGB").resize((size, size))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x)


class PowerSampler(threading.Thread):
    """Background INA219 power sampler (pip install pi-ina219)."""
    def __init__(self):
        super().__init__(daemon=True)
        from ina219 import INA219
        self.ina = INA219(shunt_ohms=0.1, max_expected_amps=3.2, busnum=1)
        self.ina.configure(voltage_range=INA219.RANGE_16V, gain=INA219.GAIN_8_320MV,
                           bus_adc=INA219.ADC_128SAMP, shunt_adc=INA219.ADC_128SAMP)
        self.samples = []
        self._run = True

    def run(self):
        while self._run:
            try:
                self.samples.append(self.ina.power())   # milliwatts
            except Exception:
                pass
            time.sleep(0.005)                            # ~200 Hz

    def stop(self):
        self._run = False

    def avg_w(self):
        if not self.samples:
            return None
        w = np.array(self.samples, dtype=float) / 1000.0
        w = w[(w > 1.0) & (w < 15.0)]      # reject dropout (~0) and spike samples
        return float(np.median(w)) if len(w) else None


def idle_power_w(seconds=4.0):
    s = PowerSampler(); s.start(); time.sleep(seconds); s.stop(); s.join(timeout=1)
    return s.avg_w()


def measure(model, x, iters, idle_w):
    import onnxruntime as ort
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(5):                                   # warmup
        sess.run(None, {name: x})
    sampler = None
    if idle_w is not None:
        sampler = PowerSampler(); sampler.start()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
    power_w = energy_mj = None
    if sampler:
        sampler.stop(); sampler.join(timeout=1)
        power_w = sampler.avg_w()
    lat = np.array(lat)
    if power_w is not None:
        net_w = max(0.0, power_w - idle_w)
        energy_mj = net_w * (lat.mean() / 1000.0) * 1000.0   # W * s -> mJ per inference
    import resource
    ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # ru_maxrss is KB on Linux
    return dict(size_mb=os.path.getsize(model) / 1e6, mean=lat.mean(), med=np.median(lat),
                p95=np.percentile(lat, 95), ram=ram_mb, power=power_w, energy=energy_mj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--ina219", action="store_true", help="measure energy via INA219")
    a = ap.parse_args()

    models = []
    for m in a.models:
        models += sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]
    x = make_input(a.img, a.img_size)

    idle_w = None
    if a.ina219:
        print("measuring idle baseline power (keep the Pi quiet ~4 s) ...", flush=True)
        idle_w = idle_power_w()
        print(f"idle power = {idle_w:.2f} W\n")

    hdr = f"{'model':26s} {'size_MB':>8s} {'mean_ms':>8s} {'med_ms':>8s} {'p95_ms':>8s} {'RAM_MB':>7s}"
    if a.ina219:
        hdr += f" {'power_W':>8s} {'E_mJ':>8s}"
    print(hdr)
    for m in models:
        r = measure(m, x, a.iters, idle_w)
        line = (f"{os.path.basename(m):26s} {r['size_mb']:8.1f} {r['mean']:8.1f} "
                f"{r['med']:8.1f} {r['p95']:8.1f} {r['ram']:7.0f}")
        if a.ina219:
            line += f" {r['power']:8.2f} {r['energy']:8.1f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
