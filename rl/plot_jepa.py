"""Publication figures for the Phase 2 JEPA encoder pretrain.

Data sources (both committed):
  runs/archive/jepa_v1/jepa_v1.csv   step, inv_train, inv_val, eff_rank
  checkpoints/jepa_pixels.pt         probe_metrics + gates

Same visual language as rl/plot_v8.py so the figure set reads as one series.

    python rl/plot_jepa.py --out-dir plots/jepa
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DPI = 130
RED, BLUE, GRAY = "#C44E52", "#4C72B0", "#bbbbbb"


def plot_training(csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    step = np.array([float(r["step"]) for r in rows])
    tr = np.array([float(r["inv_train"]) for r in rows])
    va = np.array([float(r["inv_val"]) for r in rows])
    rank = np.array([float(r["eff_rank"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(step, tr, color=BLUE, lw=1.8, label="invariance loss (train)")
    ax.plot(step, va, color=BLUE, lw=1.8, ls="--", label="invariance loss (val)")
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("JEPA invariance loss", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0, max(tr.max(), va.max()) * 1.25)

    ax2 = ax.twinx()
    ax2.plot(step, rank, color=RED, lw=2.2, label="effective rank of z")
    ax2.axhline(64, ls=":", c="gray", lw=1)
    ax2.text(step[-1], 60.5, "latent dim (64)", fontsize=8, color="gray",
             ha="right")
    ax2.set_ylabel("effective rank (collapse check)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 70)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax.set_title("phase 2: JEPA pretrain — val tracks train while latent rank "
                 "grows (no collapse, no overfit)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


def plot_probes(ckpt_path, out):
    pm = torch.load(ckpt_path, map_location="cpu")["probe_metrics"]
    names = ["hazard side", "goal side"]
    linear = [pm["hazard_acc"], pm["goal_acc"]]
    mlp = [pm["hazard_acc_mlp"], pm["goal_acc_mlp"]]
    majority = [pm["hazard_majority"], pm["goal_majority"]]

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.bar(x - w, majority, w, color=GRAY, label="majority baseline")
    ax.bar(x, linear, w, color=BLUE, alpha=0.75, label="linear probe")
    ax.bar(x + w, mlp, w, color=RED, alpha=0.75, label="MLP probe")
    for xi, vals in zip(x, zip(majority, linear, mlp)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax.text(0.02, 0.96,
            f"wall-distance regression R² = {pm['wall_r2']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("phase 2: what the frozen latent linearly encodes",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="runs/archive/jepa_v1/jepa_v1.csv")
    p.add_argument("--ckpt", default="checkpoints/jepa_pixels.pt")
    p.add_argument("--out-dir", default="plots/jepa")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_training(args.csv, f"{args.out_dir}/jepa_training.png")
    plot_probes(args.ckpt, f"{args.out_dir}/jepa_probes.png")


if __name__ == "__main__":
    main()
