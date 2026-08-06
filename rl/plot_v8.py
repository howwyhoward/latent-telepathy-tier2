"""Publication figures for race v8 (and the stage-1.5 gate that enabled it).

Same visual language as Tier 1's rl/plot_m10.py — thesis condition in red,
controls in blue/gray, per-seed dots over translucent mean bars, the `none`
floor as a dashed reference — so the two tiers' figures read as one series.

Reads the per-run JSONs written by the trainers:

  runs/race_v8/{oracle,z_t,none}[_s{2,3}].json   (rl/train_race_route.py)
  runs/route_obey_v6/cont.json                   (rl/train_route_obey.py)

Usage:
    python rl/plot_v8.py --out-dir plots
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["none", "z_t", "oracle"]
LABELS = {
    "none": "none\n(floor)",
    "z_t": "z_t\n(C1 percept)",
    "oracle": "oracle\n(ceiling)",
}
COLORS = {"none": "#bbbbbb", "z_t": "#C44E52", "oracle": "#4C72B0"}
DPI = 130


def load_runs(pattern):
    runs = defaultdict(list)
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            d = json.load(f)
        runs[d["args"]["condition"]].append(d)
    if not runs:
        raise SystemExit(f"no run JSONs matched: {pattern}")
    return runs


def seed_of(run):
    return run["args"].get("seed", 0)


# -- panel 1: learning curves -------------------------------------------------

def plot_curves(runs, out, metric="route_opt", ylabel="route-optimality",
                title="race v8: corridor choice from the message (bandit head, frozen executor)"):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    grid = np.arange(0, 6001, 128)
    for c in ORDER:
        if c not in runs:
            continue
        interp = []
        for r in runs[c]:
            ep = np.array([row["episodes"] for row in r["curve"]], dtype=float)
            y = np.array([row[metric] for row in r["curve"]], dtype=float)
            interp.append(np.interp(grid, ep, y, left=np.nan, right=y[-1]))
            ax.plot(ep, y, color=COLORS[c], alpha=0.30, lw=1.0, zorder=1)
        mean = np.nanmean(np.stack(interp), axis=0)
        ax.plot(grid, mean, color=COLORS[c], lw=2.4, zorder=2,
                label=f"{c} ({len(runs[c])} seeds)")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.text(30, 0.512, "coin flip", fontsize=8, color="gray")
    ax.set_xlabel("episodes")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, 6000)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


# -- panel 2: final bars with per-seed dots -----------------------------------

def plot_seed_bars(runs, out, key="route_opt", ylabel="final route-optimality (last 500 eps)",
                   title="race v8: message content decides the corridor", ylim=1.12,
                   fmt="{:.3f}"):
    conds = [c for c in ORDER if c in runs]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, c in enumerate(conds):
        finals = np.array([r[key] for r in runs[c]], dtype=float)
        ax.bar(i, finals.mean(), color=COLORS[c], alpha=0.45, zorder=1)
        jitter = (np.arange(len(finals)) - (len(finals) - 1) / 2) * 0.09
        ax.scatter(i + jitter, finals, color=COLORS[c], edgecolor="black",
                   linewidth=0.8, s=55, zorder=3)
        for j, (xx, yy) in enumerate(zip(i + jitter, finals)):
            ax.annotate(f"s{seed_of(runs[c][j])}", (xx, yy),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="#333333")
        ax.text(i, min(finals.mean() + 0.16, ylim - 0.04), fmt.format(finals.mean()),
                ha="center", fontsize=9)
    if "none" in runs and key == "route_opt":
        floor = float(np.mean([r[key] for r in runs["none"]]))
        ax.axhline(floor, ls="--", c="gray", lw=1, label=f"none floor ({floor:.2f})")
        ax.legend(loc="upper left")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


# -- panel 3: hazard readout --------------------------------------------------

def plot_hazard_bars(runs, out):
    conds = [c for c in ORDER if c in runs]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, c in enumerate(conds):
        vals = np.array([r["hazard"] for r in runs[c]], dtype=float)
        ax.bar(i, vals.mean(), color=COLORS[c], alpha=0.45, zorder=1)
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.09
        ax.scatter(i + jitter, vals, color=COLORS[c], edgecolor="black",
                   linewidth=0.8, s=55, zorder=3)
        ax.text(i, vals.mean() + 0.015, f"{vals.mean():.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("hazard steps / episode (lower is better)")
    ax.set_title("race v8: pre-registered readout — hazard avoidance by message content",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


# -- panel 4: decision entropy ------------------------------------------------

def plot_entropy(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grid = np.arange(0, 6001, 128)
    for c in ORDER:
        if c not in runs:
            continue
        interp = []
        for r in runs[c]:
            ep = np.array([row["episodes"] for row in r["curve"]], dtype=float)
            y = np.array([row["entropy"] for row in r["curve"]], dtype=float)
            interp.append(np.interp(grid, ep, y, left=np.nan, right=y[-1]))
            ax.plot(ep, y, color=COLORS[c], alpha=0.30, lw=1.0, zorder=1)
        mean = np.nanmean(np.stack(interp), axis=0)
        ax.plot(grid, mean, color=COLORS[c], lw=2.4, zorder=2, label=c)
    ax.axhline(np.log(2), ls=":", c="gray", lw=1)
    ax.text(30, np.log(2) + 0.012, "uniform (ln 2)", fontsize=8, color="gray")
    ax.set_xlabel("episodes")
    ax.set_ylabel("route-decision entropy (nats)")
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 0.75)
    ax.set_title("race v8: the head commits — message conditions collapse, the floor cannot",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


# -- panel 5: WP1 corruption controls ------------------------------------------

MODES = ["intact", "zero_content", "zero_all", "shuffle", "noise"]
MODE_LABELS = {
    "intact": "intact",
    "zero_content": "zero\ncontent",
    "zero_all": "zero\nwire",
    "shuffle": "shuffled\nsender",
    "noise": "gaussian\nnoise",
}


def plot_corruption(diag_glob, out):
    """Frozen v8 heads under wire corruption: greedy decisions, per-env quotas.

    Red bars = mean over z_t seeds (dots = seeds); blue diamonds = the oracle
    head. The claim in one picture: only the intact message carries the route.
    """
    z_t, oracle = defaultdict(list), {}
    for fp in sorted(glob.glob(diag_glob)):
        with open(fp) as f:
            d = json.load(f)
        cond = d["args"]["condition"]
        for m, row in d["results"].items():
            if cond == "z_t":
                z_t[m].append(row["route_opt"])
            else:
                oracle[m] = row["route_opt"]
    if not z_t:
        print(f"no corruption JSONs matched {diag_glob}; skipping")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, m in enumerate(MODES):
        vals = np.array(z_t.get(m, []), dtype=float)
        ax.bar(i, vals.mean(), color=COLORS["z_t"], alpha=0.45, zorder=1)
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.09
        ax.scatter(i + jitter, vals, color=COLORS["z_t"], edgecolor="black",
                   linewidth=0.8, s=55, zorder=3)
        ax.text(i, vals.mean() + 0.035, f"{vals.mean():.3f}", ha="center",
                fontsize=9)
        if m in oracle:
            ax.scatter(i + 0.28, oracle[m], marker="D", s=48,
                       color=COLORS["oracle"], edgecolor="black",
                       linewidth=0.8, zorder=3)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.text(len(MODES) - 0.55, 0.512, "coin flip", fontsize=8, color="gray")
    ax.scatter([], [], marker="D", color=COLORS["oracle"], edgecolor="black",
               label="oracle head")
    ax.bar(0, 0, color=COLORS["z_t"], alpha=0.45, label="z_t heads (3 seeds)")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=9)
    ax.set_ylabel("route-optimality (greedy decisions, 256 eps/mode)")
    ax.set_ylim(0, 1.12)
    ax.set_title("wire corruption: the decision lives in the message content, "
                 "and only there", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


# -- panel 6: stage-1.5 gate --------------------------------------------------

def plot_obedience(json_path, out):
    with open(json_path) as f:
        d = json.load(f)
    steps = np.array([row["global_step"] for row in d["curve"]], dtype=float) / 1e6
    fig, ax = plt.subplots(figsize=(7.5, 5))
    series = [
        ("obey_top_can", "#C44E52", "-", "obey (told: top)"),
        ("obey_bottom_can", "#4C72B0", "-", "obey (told: bottom)"),
        ("success_top_can", "#C44E52", "--", "success (told: top)"),
        ("success_bottom_can", "#4C72B0", "--", "success (told: bottom)"),
    ]
    for key, color, ls, label in series:
        y = np.array([row[key] for row in d["curve"]], dtype=float)
        ax.plot(steps, y, color=color, ls=ls, lw=1.8, label=label)
    ax.axhline(0.90, ls=":", c="gray", lw=1)
    ax.axhline(0.80, ls=":", c="gray", lw=1)
    ax.text(steps[-1], 0.905, "obedience gate 0.90", fontsize=8,
            color="gray", ha="right")
    ax.text(steps[-1], 0.755, "success gate 0.80", fontsize=8,
            color="gray", ha="right")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("rate (canonical spawns, both routes)")
    ax.set_ylim(0, 1.05)
    ax.set_title("stage 1.5: a 1-bit route command becomes obeyable from the canonical start",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/race_v8/*.json")
    p.add_argument("--obey-json", default="runs/route_obey_v6/cont.json")
    p.add_argument("--out-dir", default="plots")
    args = p.parse_args()

    import os
    race_dir = os.path.join(args.out_dir, "race_v8")
    obey_dir = os.path.join(args.out_dir, "stage15")
    os.makedirs(race_dir, exist_ok=True)
    os.makedirs(obey_dir, exist_ok=True)
    runs = load_runs(args.runs)
    for c, rs in runs.items():
        print(f"{c}: {len(rs)} seeds, route_opt {[round(r['route_opt'], 3) for r in rs]}")

    plot_curves(runs, f"{race_dir}/v8_race_curves.png")
    plot_seed_bars(runs, f"{race_dir}/v8_race_seed_bars.png")
    plot_seed_bars(
        runs, f"{race_dir}/v8_success_seed_bars.png", key="success",
        ylabel="final success (last 500 eps)",
        title="race v8: task success by message content",
    )
    plot_hazard_bars(runs, f"{race_dir}/v8_hazard_bars.png")
    plot_entropy(runs, f"{race_dir}/v8_entropy_curves.png")
    plot_corruption("runs/diag/eval_race_head_*.json",
                    f"{race_dir}/v8_corruption_bars.png")
    plot_obedience(args.obey_json, f"{obey_dir}/obey_gate_curves.png")


if __name__ == "__main__":
    main()
