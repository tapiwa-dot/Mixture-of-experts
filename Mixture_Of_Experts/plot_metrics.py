"""Turn checkpoints/metrics.jsonl into charts.

    python plot_metrics.py                     # writes assets/*.png
    python plot_metrics.py --metrics path.jsonl --out some_dir
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

TRAIN = "#2F4B9A"   # indigo
VAL = "#D9861C"     # saffron
INK = "#1D2230"
GRID = "#E3E6EE"
# Under-used experts in saffron, balanced in white, over-used in indigo.
LOAD_CMAP = LinearSegmentedColormap.from_list("load", ["#D9861C", "#F7E3C4", "#FFFFFF", "#C9D3F0", "#2F4B9A"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#5A6275",
    "ytick.color": "#5A6275",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 110,
})


def read_metrics(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def curves_figure(rows: List[Dict]):
    steps = [r["step"] for r in rows]
    tr = [(r["step"], r["train_loss"], r["train_acc"]) for r in rows if r["train_loss"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([s for s, _, _ in tr], [l for _, l, _ in tr], color=TRAIN, lw=2, label="Train")
    ax1.plot(steps, [r["val_loss"] for r in rows], color=VAL, lw=2, label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-entropy")
    ax1.legend()

    ax2.plot([s for s, _, _ in tr], [a * 100 for _, _, a in tr], color=TRAIN, lw=2, label="Train")
    ax2.plot(steps, [r["val_acc"] * 100 for r in rows], color=VAL, lw=2, label="Validation")
    ax2.set_title("Next-token accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("%")
    ax2.legend()

    best = min(rows, key=lambda r: r["val_loss"])
    ax1.scatter([best["step"]], [best["val_loss"]], color=VAL, zorder=5, s=30)
    ax1.annotate(f"best {best['val_loss']:.3f}", (best["step"], best["val_loss"]),
                 textcoords="offset points", xytext=(-10, 12), ha="right", color=INK)
    last = rows[-1]
    ax2.annotate(f"{last['val_acc']:.1%}", (last["step"], last["val_acc"] * 100),
                 textcoords="offset points", xytext=(-4, -16), ha="right", color=INK)
    fig.tight_layout()
    return fig


def load_figure(rows: List[Dict]):
    """Share of routing decisions per expert, per layer, at the last evaluation."""
    load = rows[-1]["expert_load"]
    n_layers, n_experts = len(load), len(load[0])
    fig, ax = plt.subplots(figsize=(1.2 * n_experts + 2.5, 0.6 * n_layers + 1.6))
    ax.grid(False)
    uniform = 1 / n_experts
    peak = max(max(r) for r in load)
    spread = max(peak - uniform, uniform - min(min(r) for r in load), 0.05)
    norm = TwoSlopeNorm(vcenter=uniform, vmin=max(0.0, uniform - spread), vmax=uniform + spread)
    im = ax.imshow(load, cmap=LOAD_CMAP, norm=norm, aspect="auto")
    for i in range(n_layers):
        for j in range(n_experts):
            v = load[i][j]
            strong = abs(v - uniform) > 0.7 * spread
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if strong else INK)
    ax.set_xticks(range(n_experts), [f"E{j}" for j in range(n_experts)])
    ax.set_yticks(range(n_layers), [f"Layer {i}" for i in range(n_layers)])
    ax.set_title(f"Share of tokens per expert (balanced = {uniform:.0%})")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot(metrics_path: str = "checkpoints/metrics.jsonl", out_dir: str = "assets") -> None:
    rows = read_metrics(metrics_path)
    os.makedirs(out_dir, exist_ok=True)
    for name, fig in [("training_curves.png", curves_figure(rows)), ("expert_load.png", load_figure(rows))]:
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="checkpoints/metrics.jsonl")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    plot(a.metrics, a.out)
