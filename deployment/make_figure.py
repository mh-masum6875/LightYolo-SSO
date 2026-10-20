"""Draw the Raspberry Pi 4 results figure from pi_results.csv.

    python deployment/make_figure.py            # writes pi_results.png, pi_results_dark.png, pi_results_onepager.pdf
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = pathlib.Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "pi_results.csv")))
for r in rows:
    for k in r:
        if k != "detector":
            r[k] = float(r[k])
ours = rows[0]["detector"]

THEMES = {
    "light": dict(bg="#ffffff", ink="#17202a", muted="#5a6573", grid="#e3e7eb", accent="#0f766e", other="#9aa5b1"),
    "dark": dict(bg="#0c1117", ink="#e9eef3", muted="#9ba7b5", grid="#233040", accent="#5eead4", other="#5d6b7a"),
}
PANELS = [
    ("latency_mean_ms", "Latency per inference", "ms (lower is better)", "{:.0f}"),
    ("energy_mj", "Energy per inference", "mJ (lower is better)", "{:.0f}"),
    ("peak_ram_mb", "Peak memory", "MB (lower is better)", "{:.0f}"),
    ("onnx_mb", "Model file size", "MB (lower is better)", "{:.1f}"),
]
SETUP = ("Raspberry Pi 4 Model B (4 GB), ONNX Runtime on the CPU, 100 inferences per model. "
         "Energy from an inline INA219 sensor on the 5 V rail, idle baseline subtracted.")


def ratios(key):
    base = rows[0][key]
    r = [x[key] / base for x in rows[1:]]
    return min(r), max(r)


def draw(theme, fig=None):
    t = THEMES[theme]
    fig = fig or plt.figure(figsize=(11, 7.4))
    fig.patch.set_facecolor(t["bg"])
    lo_l, hi_l = ratios("latency_mean_ms")
    lo_e, hi_e = ratios("energy_mj")
    fig.text(0.05, 0.955, "WSN-YOLO on a Raspberry Pi 4", fontsize=19, fontweight="bold", color=t["ink"], ha="left", va="top")
    fig.text(0.05, 0.905, f"{lo_l:.1f} to {hi_l:.1f} times faster and {lo_e:.1f} to {hi_e:.1f} times less energy per inference than the baselines",
             fontsize=12, color=t["muted"], ha="left", va="top")
    for i, (key, title, unit, fmt) in enumerate(PANELS):
        ax = fig.add_axes([0.13 + (i % 2) * 0.46, 0.50 - (i // 2) * 0.37, 0.30, 0.25])
        ax.set_facecolor(t["bg"])
        names = [r["detector"] for r in rows][::-1]
        vals = [r[key] for r in rows][::-1]
        cols = [t["accent"] if n == ours else t["other"] for n in names]
        bars = ax.barh(names, vals, color=cols, height=0.55)
        top = max(vals)
        ax.set_xlim(0, top * 1.22)
        for b, v, n in zip(bars, vals, names):
            ax.text(v + top * 0.02, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", ha="left", fontsize=10.5,
                    color=t["ink"], fontweight="bold" if n == ours else "normal")
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=t["ink"], pad=26)
        ax.text(0, 1.05, unit, transform=ax.transAxes, fontsize=9.5, color=t["muted"], va="bottom")
        ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=t["ink"])
        ax.tick_params(axis="x", labelsize=9, labelcolor=t["muted"], colors=t["grid"])
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
    fig.text(0.05, 0.035, SETUP, fontsize=9.5, color=t["muted"], ha="left", va="bottom", wrap=True)
    return fig


for theme, name in (("light", "pi_results.png"), ("dark", "pi_results_dark.png")):
    f = draw(theme)
    f.savefig(HERE / name, dpi=170, facecolor=f.get_facecolor())
    plt.close(f)

# one page PDF: the figure plus a short written summary
with PdfPages(HERE / "pi_results_onepager.pdf") as pdf:
    f = plt.figure(figsize=(8.27, 11.69))
    f.patch.set_facecolor("#ffffff")
    inner = draw("light", plt.figure(figsize=(11, 7.4)))
    inner.savefig(HERE / "_tmp.png", dpi=150, facecolor="#ffffff")
    plt.close(inner)
    img = plt.imread(HERE / "_tmp.png")
    ax = f.add_axes([0.04, 0.50, 0.92, 0.46]); ax.imshow(img); ax.axis("off")
    r = {x["detector"]: x for x in rows}; w = r[ours]
    lines = [
        ("Summary", True),
        (f"WSN-YOLO runs at {w['latency_mean_ms']:.1f} ms per inference on a Raspberry Pi 4, in a {w['onnx_mb']:.1f} MB model that uses {w['peak_ram_mb']:.0f} MB of memory.", False),
        (f"Each inference costs {w['energy_mj']:.0f} mJ at {w['power_w']:.2f} W. YOLOv5n needs {r['YOLOv5n']['energy_mj']:.0f} mJ, YOLOv8n {r['YOLOv8n']['energy_mj']:.0f} mJ and YOLOv7-tiny {r['YOLOv7-tiny']['energy_mj']:.0f} mJ.", False),
        ("Power draw is similar for all four models (4.8 to 5.2 W), so the energy gap comes mainly from how long each model keeps the processor busy.", False),
        ("", False),
        ("Method", True),
        (SETUP, False),
        ("Latency is the mean over the runs, with the 95th percentile within 2 ms of it for every model. Numbers are in pi_results.csv.", False),
    ]
    y = 0.44
    import textwrap
    for text, head in lines:
        for ln in (textwrap.wrap(text, 92) or [""]):
            f.text(0.07, y, ln, fontsize=14 if head else 11, fontweight="bold" if head else "normal", color="#17202a", va="top")
            y -= 0.034 if head else 0.025
        y -= 0.006
    pdf.savefig(f, facecolor="#ffffff"); plt.close(f)
(HERE / "_tmp.png").unlink()
print("wrote pi_results.png, pi_results_dark.png, pi_results_onepager.pdf")
