#!/usr/bin/env python3
"""Generate paper-ready figures from post-processing-only results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
results_path = ROOT / "data" / "outputs" / "results.json"
out_dir = ROOT / "data" / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(results_path.read_text())
metrics = data["metrics"]
dataset = data["dataset"]
sensitivity = data["sensitivity"]

MODELS = ["baseline", "mitigation_a", "mitigation_b", "mitigation_c_constrained"]
LABELS = [
    "Baseline",
    "Mitigation A\n(+12% boost)",
    "Mitigation B\n(+28% boost)",
    "Mitigation C\n(constrained)",
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Figure 1: Exposure male share (post-processing baselines only) ───────────
fig, ax = plt.subplots(figsize=(8, 4.5))
shares = [metrics[m]["exposure_male_share"] for m in MODELS]
bars = ax.bar(LABELS, shares, color=COLORS, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
train_share = dataset["train_male_share_among_labeled"]
if train_share is not None:
    ax.axhline(train_share, color="gray", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(
        3.45,
        float(train_share) + 0.01,
        f"Training data\n{train_share:.1%}",
        color="gray",
        fontsize=8,
        va="bottom",
        ha="right",
    )
ax.axhline(0.5, color="black", linestyle=":", linewidth=1, zorder=2)
ax.text(3.45, 0.505, "Parity (50%)", fontsize=8.5, color="black", ha="right")
for bar, val in zip(bars, shares):
    if val is None:
        continue
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        float(val) + 0.008,
        f"{float(val):.1%}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.set_ylabel("Male share of position-weighted exposure")
ax.set_ylim(0, 0.85)
ax.set_title(
    "Provider-side gender exposure: post-processing mitigations",
    fontsize=11,
    pad=10,
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(out_dir / "fig1_exposure_male_share.png", dpi=180)
plt.close(fig)
print("Saved fig1_exposure_male_share.png")

# ── Figure 2: Utility metrics ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
metric_keys = ["precision_at_k", "recall_at_k", "ndcg_at_k"]
metric_names = ["Precision@10", "Recall@10", "NDCG@10"]
x = np.arange(len(metric_names))
width = 0.18
for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
    vals = [metrics[model][k] for k in metric_keys]
    offset = (i - 1.5) * width
    rects = ax.bar(x + offset, vals, width, label=label.replace("\n", " "), color=color, zorder=3)
    for rect, v in zip(rects, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() - 0.008,
            f"{v:.3f}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 0.22)
ax.set_title("Recommendation quality: baseline vs. mitigations", fontsize=11, pad=10)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(out_dir / "fig2_utility_metrics.png", dpi=180)
plt.close(fig)
print("Saved fig2_utility_metrics.png")

# ── Figure 3: Fairness–utility (female score boost sweep) ───────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ms = sensitivity["exposure_male_share"]
nd = sensitivity["ndcg_at_k"]
ax.plot(ms, nd, "o--", color="#4C72B0", linewidth=1.8, markersize=6, zorder=3, alpha=0.85)
ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(0.502, min(nd) + 0.0003, "Parity (50%)", fontsize=8.5, color="black")
ax.set_xlabel("Male share of position-weighted exposure (← fairer)", fontsize=10)
ax.set_ylabel("NDCG@10", fontsize=10)
ax.set_title("Fairness–utility trade-off: post-processing boost sweep", fontsize=11, pad=10)
ax.grid(alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(out_dir / "fig3_fairness_utility_tradeoff.png", dpi=180)
plt.close(fig)
print("Saved fig3_fairness_utility_tradeoff.png")

# ── Figure 4: Training bias vs recommendations ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
stages = ["Training\nData", "Baseline\nRecs", "Mitigation A\nRecs", "Mitigation B\nRecs"]
vals = [
    dataset["train_male_share_among_labeled"],
    metrics["baseline"]["exposure_male_share"],
    metrics["mitigation_a"]["exposure_male_share"],
    metrics["mitigation_b"]["exposure_male_share"],
]
colors = ["#888888", "#4C72B0", "#55A868", "#C44E52"]
bars = ax.bar(stages, vals, color=colors, width=0.5, zorder=3)
ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
ax.text(3.4, 0.51, "Parity", fontsize=9)
for bar, val in zip(bars, vals):
    if val is None:
        continue
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        float(val) + 0.008,
        f"{float(val):.1%}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.set_ylabel("Male share of exposure")
ax.set_ylim(0, 0.85)
ax.set_title(
    "From training data to recommendations:\ngender exposure at each stage",
    fontsize=11,
    pad=10,
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(out_dir / "fig4_amplification.png", dpi=180)
plt.close(fig)
print("Saved fig4_amplification.png")

print(f"\nAll figures saved to {out_dir}")
