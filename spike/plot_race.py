"""Race comparison plot: success + hazard-steps curves per condition.

Reads every runs/race/*.json written by rl/train_race.py, averages curves
across seeds within a condition, and writes a two-panel figure plus a
final-metrics table to stdout.

    python spike/plot_race.py --run_dir runs/race --out runs/race/race.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["none", "position", "z_t", "raw"]
COLORS = {"none": "#888888", "position": "#1f77b4", "z_t": "#d62728", "raw": "#2ca02c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="runs/race")
    p.add_argument("--out", type=str, default="runs/race/race.png")
    args = p.parse_args()

    runs = defaultdict(list)
    for f in sorted(Path(args.run_dir).glob("*.json")):
        with open(f) as fh:
            r = json.load(fh)
        runs[r["condition"]].append(r)

    if not runs:
        print(f"no run jsons in {args.run_dir}")
        return

    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
    print(f"{'condition':>10} {'seeds':>5} {'success':>8} {'hazard/ep':>10}")
    for cond in ORDER:
        if cond not in runs:
            continue
        curves = [np.array(r["curve"]) for r in runs[cond]]
        # truncate to shortest run so seeds average cleanly
        n = min(len(c) for c in curves)
        arr = np.stack([c[:n] for c in curves])  # (seeds, iters, 5)
        steps = arr[0, :, 0]
        succ_m = arr[:, :, 1].mean(0)
        haz_m = arr[:, :, 4].mean(0)
        c = COLORS[cond]
        ax_s.plot(steps, succ_m, color=c, label=cond)
        ax_h.plot(steps, haz_m, color=c, label=cond)
        if arr.shape[0] > 1:
            ax_s.fill_between(steps, arr[:, :, 1].min(0), arr[:, :, 1].max(0),
                              color=c, alpha=0.15)
            ax_h.fill_between(steps, arr[:, :, 4].min(0), arr[:, :, 4].max(0),
                              color=c, alpha=0.15)
        fs = np.mean([r["final_success"] for r in runs[cond]])
        fh_ = np.mean([r["final_hazard_steps"] for r in runs[cond]])
        print(f"{cond:>10} {arr.shape[0]:>5} {fs:>8.3f} {fh_:>10.2f}")

    ax_s.set_xlabel("env steps"); ax_s.set_ylabel("success (last 100 eps)")
    ax_s.set_title("Success"); ax_s.legend(); ax_s.grid(alpha=0.3)
    ax_h.set_xlabel("env steps"); ax_h.set_ylabel("hazard steps / episode")
    ax_h.set_title("Hazard exposure (pre-registered readout)")
    ax_h.legend(); ax_h.grid(alpha=0.3)
    fig.suptitle("Chokepoint race: message conditions")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
