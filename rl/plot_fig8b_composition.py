"""Figure 8b — the composition check: the whole pipeline, with the probe supervised.

The decision this figure records: before asking whether *reward* can recruit the
message (race v8), check that every other link in the chain already works. Hand
the route decision to a supervised logistic probe on the frozen latent, keep
everything else frozen, and measure end to end. It composes at ceiling, which
narrows six generations of nulls to exactly one open question: can reward
replace the probe's labels?

So the figure has to make two things obvious at a glance:
  1. the chain is frozen everywhere except one box, and
  2. that one box is the only thing race v8 has to earn.

Every number is parsed from the committed eval log, not from the report prose:

    runs/route_obey_v6/eval_pixels_to_route.log

Usage:
    python rl/plot_fig8b_composition.py --out-dir plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

DPI = 130
RED = "#C44E52"        # the unearned link, and the thesis colour throughout
BLUE = "#4C72B0"
GREEN = "#55A868"
PURPLE = "#8172B2"
DARK = "#333333"
SLATE = "#6b7280"
EDGE = "#9aa2ac"
FILL = "#eef1f5"


def parse_eval(path: Path) -> dict:
    """Pull the composition numbers out of the eval log."""
    text = path.read_text(errors="ignore")

    def grab(pattern, cast=float, what=""):
        m = re.search(pattern, text, re.M)
        if not m:
            raise SystemExit(f"{path}: could not find {what or pattern}")
        return tuple(cast(g) for g in m.groups()) if len(m.groups()) > 1 \
            else cast(m.group(1))

    n_probe, train_acc, probe_acc = grab(
        r"probe:\s+(\d+) samples\s+train acc\s+([\d.]+)\s+held-out acc\s+([\d.]+)",
        float, "the probe line")
    ref_top, ref_bottom = grab(
        r"ground-truth route read\s+([\d.]+)/([\d.]+)\s+success", float,
        "the ground-truth reference")
    return dict(
        n_probe=int(n_probe), train_acc=train_acc, probe_acc=probe_acc,
        n_eps=int(grab(r"route -> navigation \((\d+) episodes\)", float, "episodes")),
        decode=grab(r"decode accuracy\s+([\d.]+)", float, "decode accuracy"),
        success=grab(r"^success\s+([\d.]+)", float, "success"),
        obeyed=grab(r"obeyed decode\s+([\d.]+)", float, "obeyed decode"),
        hazard=grab(r"hazard steps\s+([\d.]+) per episode", float, "hazard steps"),
        ref_top=ref_top, ref_bottom=ref_bottom,
        blind_hazard=grab(r"pays ~(\d+) hazard steps", float, "the blind reference"),
    )


def _pill(ax, x, y, text, color, fc):
    ax.text(x, y, text, ha="center", va="center", fontsize=8.6, color=color,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.38", fc=fc, ec=color, lw=1.2))


def draw_pipeline(ax, m):
    """The chain itself: six stages, one of them not frozen."""
    stages = [
        ("scout pixels", "the only view\nof the slab", FILL, EDGE, DARK, None),
        ("frozen JEPA\nencoder", "Phase 2 checkpoint", FILL, EDGE, DARK, "FROZEN"),
        ("logistic probe", "64-D → 2,\ntrained on labels", "#fbeced", RED, RED,
         "SUPERVISED"),
        ("route", "1 bit:\ntop or bottom", "#f1eef8", PURPLE, PURPLE, None),
        ("frozen executor", "v6 cont.pt,\nroute-conditioned", FILL, EDGE, DARK,
         "FROZEN"),
        ("navigation", f"{m['n_eps']} episodes", "#edf4ef", GREEN, DARK, None),
    ]
    flows = ["64×64×3", "z_t (64-D)", "argmax", "conditions", "cmd_vel"]
    w, gap, y0, y1 = 12.2, 4.9, 55.0, 78.0
    yc = (y0 + y1) / 2
    xs = [1.2 + i * (w + gap) for i in range(len(stages))]

    for x, (title, sub, fc, ec, tc, pill) in zip(xs, stages):
        ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=2.2 if pill == "SUPERVISED" else 1.5,
                                    zorder=3, mutation_aspect=0.42))
        ax.text(x + w / 2, yc + 3.6, title, ha="center", va="center", fontsize=11.4,
                color=tc, fontweight="bold", zorder=5, linespacing=1.35)
        ax.text(x + w / 2, yc - 4.6, sub, ha="center", va="center", fontsize=8.9,
                color=SLATE if tc is DARK else tc, zorder=5, linespacing=1.4)
        if pill:
            _pill(ax, x + w / 2, y1, pill, RED if pill == "SUPERVISED" else SLATE,
                  "#fbeced" if pill == "SUPERVISED" else "white")

    for x, flow in zip(xs[:-1], flows):
        ax.annotate("", xy=(x + w + gap - 0.6, yc), xytext=(x + w + 0.6, yc),
                    arrowprops=dict(arrowstyle="-|>", color="#8b939e", lw=2.0,
                                    mutation_scale=17), zorder=4)
        ax.text(x + w + gap / 2, yc + 4.4, flow, ha="center", va="bottom",
                fontsize=8.4, color=SLATE, style="italic", zorder=5)

    # what was measured where, in one lane so the three chips line up
    chips = [
        (2, f"held-out accuracy\n{m['probe_acc']:.3f}", RED),
        (3, f"decoded in the loop\n{m['decode']:.3f}", PURPLE),
        (5, f"success {m['success']:.3f}\nhazard {m['hazard']:.2f} / episode", GREEN),
    ]
    for i, text, color in chips:
        cx = xs[i] + w / 2
        ax.plot([cx, cx], [y0 - 1.0, 49.0], color=color, lw=1.2, ls=(0, (2, 2)),
                zorder=2)
        ax.text(cx, 43.5, text, ha="center", va="center", fontsize=9.6, color=color,
                fontweight="bold", zorder=5, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=color, lw=1.2))

    ax.annotate("the one link that is not earned —\n"
                "race v8 replaces it with reward",
                xy=(xs[2] - 0.4, yc - 5.5), xytext=(14.5, 21.0),
                ha="center", va="center", fontsize=10.4, color=RED,
                fontweight="bold", zorder=6, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fbeced", ec=RED, lw=1.4),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                mutation_scale=16,
                                connectionstyle="arc3,rad=-0.14"))
    ax.text(66.0, 21.0,
            f"every link measured at ceiling:   probe {m['probe_acc']:.3f}   ·   "
            f"decode {m['decode']:.3f}   ·   obeyed {m['obeyed']:.3f}",
            ha="center", va="center", fontsize=10.4, color=GREEN,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#edf4ef", ec=GREEN, lw=1.4))

    ax.set_xlim(0, 100)
    ax.set_ylim(12, 90)
    ax.axis("off")


def draw_success(ax, m):
    rows = [
        (m["success"], f"route decoded from the scout's pixels  —  this pipeline, "
                       f"{m['n_eps']} episodes", GREEN, True),
        (m["ref_top"], "reference: route handed to the executor, slab top", SLATE,
         False),
        (m["ref_bottom"], "reference: route handed to the executor, slab bottom",
         SLATE, False),
    ]
    for i, (v, label, color, thesis) in enumerate(rows):
        y = -i
        ax.barh(y, v, 0.36, color=color, alpha=1.0 if thesis else 0.50,
                ec="white", zorder=3)
        ax.text(v - 0.012, y, f"{v:.3f}", va="center", ha="right", fontsize=11.0,
                color="white", fontweight="bold", zorder=4)
        ax.text(0.006, y + 0.30, label, fontsize=9.9, color=DARK, ha="left",
                va="bottom", fontweight="bold" if thesis else "normal", zorder=4)
    ax.axvline(1.0, color=DARK, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(1.012, -2.0, "every\nepisode", fontsize=9.4, color=DARK, ha="left",
            va="center", linespacing=1.4)
    ax.set_xlim(0, 1.19)
    ax.set_ylim(-2.62, 0.72)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks([])
    ax.set_xlabel("success rate", fontsize=10.6, x=0.42)
    ax.grid(alpha=0.20, lw=0.6, axis="x")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def draw_hazard(ax, m):
    rows = [(m["hazard"], "this pipeline", GREEN),
            (m["blind_hazard"], "reference: a blind policy, no route information",
             RED)]
    for i, (v, label, color) in enumerate(rows):
        y = -0.85 * i
        if v > 0:
            ax.barh(y, v, 0.36, color=color, alpha=0.85, ec="white", zorder=3)
            ax.text(v - 0.4, y, f"~{v:.0f}", va="center", ha="right", fontsize=11.0,
                    color="white", fontweight="bold", zorder=4)
        else:   # a zero bar still has to read as a measurement, not a gap
            ax.plot([0, 0], [y - 0.18, y + 0.18], color=color, lw=3.0,
                    solid_capstyle="butt", zorder=3)
            ax.text(0.32, y, f"{v:.2f}", va="center", ha="left", fontsize=11.0,
                    color=color, fontweight="bold", zorder=4)
        ax.text(0.16, y + 0.30, label, fontsize=9.9, color=DARK, ha="left",
                va="bottom", fontweight="bold" if v == 0 else "normal", zorder=4)
    ax.set_xlim(0, 25)
    ax.set_ylim(-1.36, 0.61)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([])
    ax.set_xlabel("hazard steps per episode", fontsize=10.6, x=0.40)
    ax.grid(alpha=0.20, lw=0.6, axis="x")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = ap.parse_args()

    log = args.runs_dir / "route_obey_v6" / "eval_pixels_to_route.log"
    m = parse_eval(log)
    out_dir = args.out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig8b_composition_check.png"

    fig = plt.figure(figsize=(16.4, 8.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.20, 0.80],
                          width_ratios=[1.42, 1.0],
                          left=0.032, right=0.988, top=0.885, bottom=0.150,
                          hspace=0.30, wspace=0.16)
    ax_pipe = fig.add_subplot(gs[0, :])
    draw_pipeline(ax_pipe, m)
    ax_s = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])
    draw_success(ax_s, m)
    draw_hazard(ax_h, m)

    for ax, step in ((ax_pipe, "1.  the pipeline, end to end"),
                     (ax_s, "2.  did it reach the goal?"),
                     (ax_h, "3.  did it avoid the hazard?")):
        fig.text(ax.get_position().x0, ax.get_position().y1 + 0.014, step,
                 fontsize=13.0, ha="left", va="bottom", fontweight="bold",
                 color=DARK)

    fig.suptitle("Figure 8b — the pipeline composes: with the route supervised, "
                 f"pixels to navigation succeeds {m['success'] * m['n_eps']:.0f} of "
                 f"{m['n_eps']} times", fontsize=16.0, y=0.965)
    fig.text(0.032, 0.012,
             f"Source: runs/route_obey_v6/eval_pixels_to_route.log. Probe trained and "
             f"held out on {m['n_probe']} samples (train {m['train_acc']:.3f}); "
             f"closed-loop numbers over {m['n_eps']} episodes, route decoded at step 2 "
             f"because reset-time frames are still stale.\nThe grey bars are the "
             f"Stage 1.5 obedience gate for the same executor with the route handed to "
             f"it directly — canonical spawns, a different episode set,\nshown for "
             f"scale and not as a paired comparison. The blind figure is the no-comms "
             f"hazard cost quoted in the same log.",
             ha="left", va="bottom", fontsize=8.8, color="#828a95", linespacing=1.6)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    for k, v in m.items():
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
