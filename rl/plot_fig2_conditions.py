"""Figure 2 — one channel, six message contents (Tier 2).

Tier 1's Figure 3 redrawn for the pixel substrate, same visual language
(rl/plot_figure3_conditions.py in ~/latent-telepathy). Left: the
content-controlled channel, with the shared 66-float wire drawn explicitly as
2 anchor floats + 64 content floats. Right: the six-condition ladder ordered
by information content, floor at the bottom, radio-infeasible ceiling at the
top, thesis conditions in red. The oracle hangs below the rule — it is not a
rung, it is the diagnostic that asks whether the optimizer works at all.

Difference from Tier 1's version: every row has been raced, so each carries
its measured route-optimality. Those markers are read from the run JSONs
(same pooling as rl/plot_v8.py's sweep), never typed in.

The row-6 payload swatch is the scout's actual camera frame at the encoder's
64x64 input resolution — literally what that condition puts on the wire.

    python rl/plot_fig2_conditions.py
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

TEXT_DARK = "#20222a"
NOTE_GRAY = "#6a707a"
CHIP_FACE, CHIP_EDGE = "#f5f6f8", "#5b6472"
BASE_GRAY = "#8a919c"
THESIS_RED = "#c1292e"
CEIL_BLUE = "#2f5f8f"
LATENT_PURPLE = "#5b5fc7"
ANCHOR_SLATE = "#7d8794"
SENDER_RED = "#c1292e"
RECEIVER_BLUE = "#2f5f8f"
TRACK_FACE = "#e9ebef"

BADGE_COLOR = {"gray": BASE_GRAY, "thesis": THESIS_RED, "ceiling": CEIL_BLUE}

ROWS = [
    dict(n=1, cond="none", name="Floor", role="content-free wire",
         badge="gray", kind="zeros", wire="66-d", tag=None),
    dict(n=2, cond="position", name="Position", role="standard baseline",
         badge="gray", kind="pos", wire="66-d", tag="constant wire"),
    dict(n=3, cond="kinematic", name="Kinematic", role="steelman message",
         badge="gray", kind="traj", wire="66-d", tag="constant wire"),
    dict(n=4, cond="z_t", name="Ours \u2014 C1", role="perception",
         badge="thesis", kind="latent", wire="66-d", tag=None),
    dict(n=5, cond="z_hat", name="Ours \u2014 C2", role="prediction",
         badge="thesis", kind="latent_pred", wire="66-d", tag=None),
    dict(n=6, cond="raw_obs", name="Ceiling", role="radio-infeasible",
         badge="ceiling", kind="patch", wire="12,290-d", tag="186\u00d7 wider"),
]
ORACLE = dict(cond="oracle", name="Oracle", role="off-ladder diagnostic",
              kind="bit", wire="66-d", tag=None)


def load_results():
    """Same pooling as plot_v8.py's sweep: v8b (all conditions, anchored floor,
    headline seeds 4-5) plus v8's z_t/oracle seeds 1-3."""
    runs = defaultdict(list)
    for pat in ("runs/race_v8b/*.json", "runs/race_v8/z_t*.json",
                "runs/race_v8/oracle*.json"):
        for fp in sorted(glob.glob(pat)):
            with open(fp) as f:
                d = json.load(f)
            runs[d["args"]["condition"]].append(float(d["route_opt"]))
    return {c: sorted(v) for c, v in runs.items()}


# --- shared glyphs ---------------------------------------------------------

def chip(ax, x, y, label, *, w=1.25, h=0.56, fontsize=8.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=CHIP_FACE, edgecolor=CHIP_EDGE,
                                linewidth=1.1, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=TEXT_DARK, family="monospace", zorder=6)


def latent_bars(ax, x, y, *, w=1.15, seed=0, color=LATENT_PURPLE):
    rng = np.random.default_rng(seed)
    heights = rng.uniform(0.08, 0.34, size=11)
    bw = w / len(heights)
    for i, hgt in enumerate(heights):
        ax.add_patch(Rectangle((x - w / 2 + i * bw + bw * 0.12, y - 0.19),
                               bw * 0.76, hgt, facecolor=color,
                               edgecolor="none", zorder=6))


def clock_icon(ax, x, y, *, r=0.155, color=THESIS_RED):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                        linewidth=1.3, zorder=7))
    ax.plot([x, x], [y, y + r * 0.55], color=color, lw=1.2, zorder=8,
            solid_capstyle="round")
    ax.plot([x, x + r * 0.45], [y, y], color=color, lw=1.2, zorder=8,
            solid_capstyle="round")


def trajectory_glyph(ax, x, y, *, w=1.15, color=NOTE_GRAY):
    ax.plot([x - w / 2, x - 0.1], [y - 0.09, y + 0.1], color=color, lw=1.6, zorder=6)
    ax.add_patch(FancyArrowPatch((x - 0.14, y + 0.08), (x + w / 2 - 0.24, y - 0.05),
                                 arrowstyle="-|>", mutation_scale=9, color=color,
                                 linewidth=1.6, zorder=6))
    for dx in (0.02, 0.18, 0.34):
        ax.plot(x + w / 2 - 0.2 + dx, y - 0.06 - dx * 0.2, "o", color=color,
                markersize=2.2, zorder=6)


def zeros_glyph(ax, x, y, *, w=1.25):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.2), w, 0.4,
                                boxstyle="round,pad=0.0,rounding_size=0.07",
                                facecolor="#eceef1", edgecolor="#c7cbd1",
                                linewidth=1.0, hatch="////", zorder=5))
    ax.text(x, y, "0 \u2026 0", ha="center", va="center", fontsize=9.2,
            color="#8a919c", family="monospace", fontweight="bold", zorder=6)


def position_glyph(ax, x, y, *, color=NOTE_GRAY):
    ax.plot([x - 0.42, x + 0.42], [y - 0.16, y - 0.16], color=color, lw=1.0, zorder=6)
    ax.plot([x - 0.32, x - 0.32], [y - 0.16, y + 0.2], color=color, lw=1.0, zorder=6)
    ax.plot(x + 0.06, y + 0.02, "o", color=color, markersize=4.5, zorder=6)
    ax.text(x + 0.28, y + 0.03, "(x, y)", ha="left", va="center", fontsize=8.6,
            color=TEXT_DARK, family="monospace", zorder=6)


def bit_glyph(ax, x, y, *, color=ANCHOR_SLATE):
    for i, filled in enumerate((True, False)):
        ax.add_patch(Rectangle((x - 0.34 + i * 0.34, y - 0.15), 0.3, 0.3,
                               facecolor=color if filled else "white",
                               edgecolor=color, linewidth=1.2, zorder=6))
    ax.text(x + 0.5, y, "1 bit", ha="left", va="center", fontsize=8.6,
            color=TEXT_DARK, family="monospace", zorder=6)


def frame_swatch(ax, x, y, path, *, size=0.58):
    """Row 6's payload drawn as the scout's real frame at the encoder's 64x64
    input resolution — the literal contents of that wire."""
    img = plt.imread(path)
    n = 64
    step = max(1, img.shape[0] // n)
    small = img[::step, ::step][:n, :n]
    half = size / 2
    ax.imshow(small, extent=[x - half, x + half, y - half, y + half],
              aspect="auto", zorder=6, interpolation="nearest")
    ax.add_patch(Rectangle((x - half, y - half), size, size, facecolor="none",
                           edgecolor=CEIL_BLUE, linewidth=1.0, zorder=7))


# --- left panel: the channel ----------------------------------------------

def draw_channel_panel(ax):
    cx_s, cx_r, cy = 1.35, 4.75, 5.05
    ax.add_patch(FancyBboxPatch((cx_s + 0.6, cy - 0.2), cx_r - cx_s - 1.2, 0.4,
                                boxstyle="round,pad=0.0,rounding_size=0.16",
                                facecolor="#e4e6ea", edgecolor="#9aa1ab",
                                linewidth=1.3, zorder=3))
    ax.add_patch(FancyArrowPatch((cx_s + 0.75, cy), (cx_r - 0.72, cy),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#7f8794", linewidth=1.3, zorder=4))
    for cx, color, name, sub in ((cx_s, SENDER_RED, "scout", "sender"),
                                 (cx_r, RECEIVER_BLUE, "route head", "receiver")):
        ax.add_patch(Circle((cx, cy), 0.46, facecolor=color, edgecolor="white",
                            linewidth=2.0, zorder=5))
        ax.text(cx, cy + 0.86, name, ha="center", va="center", fontsize=12.0,
                fontweight="bold", color=color)
        ax.text(cx, cy + 0.52, sub, ha="center", va="center", fontsize=8.6,
                color=NOTE_GRAY, style="italic")
    ax.text(cx_r, cy - 0.74, "4,483 trainable params on a 66-d wire",
            ha="center", va="center", fontsize=8.4, color=NOTE_GRAY, style="italic")

    # the wire, drawn to scale in structure: 2 anchor floats + 64 content floats
    wx0, wx1, wy, wh = 0.55, 5.55, cy - 1.65, 0.42
    frac_anchor = 2 / 66
    xa = wx0 + (wx1 - wx0) * frac_anchor
    ax.add_patch(Rectangle((wx0, wy - wh / 2), xa - wx0, wh,
                           facecolor=ANCHOR_SLATE, edgecolor="white",
                           linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((xa, wy - wh / 2), wx1 - xa, wh,
                           facecolor=LATENT_PURPLE, alpha=0.85, edgecolor="none",
                           zorder=5))
    for i in range(1, 8):
        tx = xa + (wx1 - xa) * i / 8
        ax.plot([tx, tx], [wy - wh / 2, wy + wh / 2], color="white", lw=0.9, zorder=6)
    ax.add_patch(Rectangle((wx0, wy - wh / 2), wx1 - wx0, wh, facecolor="none",
                           edgecolor=CHIP_EDGE, linewidth=1.1, zorder=7))

    ax.plot([wx0, wx0, xa, xa], [wy - wh / 2 - 0.12, wy - wh / 2 - 0.26,
                                 wy - wh / 2 - 0.26, wy - wh / 2 - 0.12],
            color=NOTE_GRAY, lw=1.0, zorder=5)
    ax.plot([xa, xa, wx1, wx1], [wy - wh / 2 - 0.12, wy - wh / 2 - 0.26,
                                 wy - wh / 2 - 0.26, wy - wh / 2 - 0.12],
            color=NOTE_GRAY, lw=1.0, zorder=5)
    ax.text(wx0 - 0.05, wy - wh / 2 - 0.36, "2 anchor\n(\u0394pos / r)",
            ha="left", va="top", fontsize=8.2, color=TEXT_DARK, linespacing=1.25)
    ax.text((xa + wx1) / 2 + 0.35, wy - wh / 2 - 0.36, "64 content floats",
            ha="center", va="top", fontsize=9.0, color=TEXT_DARK)
    ax.text((wx0 + wx1) / 2, wy + wh / 2 + 0.2, "one 66-float wire, shared by every condition",
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=TEXT_DARK)

    labels = ["same 66-float wire", "same anchored delivery", "same frozen executor"]
    top = wy - 1.75
    ax.text((wx0 + wx1) / 2, top + 0.45, "held fixed", ha="center", va="center",
            fontsize=8.8, color=NOTE_GRAY, style="italic")
    for i, lab in enumerate(labels):
        ly = top - i * 0.66
        ax.add_patch(FancyBboxPatch((wx0 + 0.35, ly - 0.23), (wx1 - wx0) - 0.7, 0.46,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=CHIP_FACE, edgecolor=CHIP_EDGE,
                                    linewidth=1.0, zorder=4))
        ax.text((wx0 + wx1) / 2, ly, lab, ha="center", va="center", fontsize=10.2,
                fontweight="bold", color=TEXT_DARK, zorder=5)

    ax.text((wx0 + wx1) / 2, cy + 1.6, "Content-controlled\ncommunication experiment",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=TEXT_DARK, linespacing=1.3)
    ax.text((wx0 + wx1) / 2, top - 2 * 0.66 - 0.72,
            "only the message content is\nthe independent variable",
            ha="center", va="center", fontsize=9.8, color=NOTE_GRAY,
            style="italic", linespacing=1.35)
    ax.text((wx0 + wx1) / 2, top - 2 * 0.66 - 1.34,
            "row 6 breaks the width on purpose \u2014\nthat is what makes it a ceiling",
            ha="center", va="center", fontsize=8.4, color=CEIL_BLUE,
            style="italic", linespacing=1.3)

    ax.set_xlim(0.3, 5.85)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.set_aspect("equal")
    ax.axis("off")


# --- right panel: the ladder ----------------------------------------------

X_ARROW, X_BADGE, X_NAME = 0.15, 1.0, 2.05
BOX_X0, BOX_X1 = 4.45, 8.75
X_ANCHOR, X_PAYLOAD = 5.18, 7.0
X_WIRE = 8.9
BAR_X0, BAR_W = 10.15, 1.3

# both panels share one vertical scale so aspect="equal" renders them at the
# same size; the ladder sets it
Y_BOTTOM, Y_TOP = -1.35, 6.9


def result_marker(ax, y, vals, color):
    mean = float(np.mean(vals))
    ax.add_patch(Rectangle((BAR_X0, y - 0.11), BAR_W, 0.22, facecolor=TRACK_FACE,
                           edgecolor="#d3d7dd", linewidth=0.7, zorder=4))
    ax.add_patch(Rectangle((BAR_X0, y - 0.11), BAR_W * mean, 0.22, facecolor=color,
                           edgecolor="none", zorder=5))
    ax.plot([BAR_X0 + BAR_W * 0.5] * 2, [y - 0.15, y + 0.15], color="#8a919c",
            lw=0.9, ls=(0, (2, 1.6)), zorder=6)
    star = "*" if max(vals) - min(vals) > 0.2 else ""  # flags the split seeds
    ax.text(BAR_X0 + BAR_W + 0.16, y + 0.045, f"{mean:.3f}{star}", ha="left",
            va="center", fontsize=10.4, fontweight="bold", color=color, zorder=6)
    ax.text(BAR_X0 + BAR_W + 0.16, y - 0.24, f"{len(vals)} seeds", ha="left",
            va="center", fontsize=7.8, color=NOTE_GRAY, zorder=6)


def draw_row(ax, y, spec, vals, *, dashed=False):
    accent = {"gray": BASE_GRAY, "thesis": THESIS_RED,
              "ceiling": CEIL_BLUE}.get(spec.get("badge"), ANCHOR_SLATE)
    name_color = accent if spec.get("badge") in ("thesis", "ceiling") else TEXT_DARK
    if dashed:
        name_color = ANCHOR_SLATE

    if "n" in spec:
        ax.add_patch(Circle((X_BADGE, y), 0.28, facecolor=accent, edgecolor="white",
                            linewidth=1.6, zorder=6))
        ax.text(X_BADGE, y - 0.01, str(spec["n"]), ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="white", zorder=7)

    ax.text(X_NAME, y + 0.17, spec["name"], ha="left", va="center", fontsize=12.2,
            fontweight="bold", color=name_color, zorder=6)
    ax.text(X_NAME, y - 0.19, spec["role"], ha="left", va="center", fontsize=9.0,
            color=NOTE_GRAY, style="italic", zorder=6)

    ax.add_patch(FancyBboxPatch((BOX_X0, y - 0.36), BOX_X1 - BOX_X0, 0.72,
                                boxstyle="round,pad=0.02,rounding_size=0.11",
                                facecolor="white", edgecolor=accent if not dashed else "#c7cbd1",
                                linewidth=1.4, linestyle=(0, (4, 2)) if dashed else "solid",
                                zorder=4))
    chip(ax, X_ANCHOR, y, "anchor", w=1.2)

    kind = spec["kind"]
    if kind == "zeros":
        zeros_glyph(ax, X_PAYLOAD, y)
    elif kind == "pos":
        position_glyph(ax, X_PAYLOAD, y)
    elif kind == "traj":
        trajectory_glyph(ax, X_PAYLOAD, y)
    elif kind == "latent":
        latent_bars(ax, X_PAYLOAD - 0.28, y, seed=1)
        ax.text(X_PAYLOAD + 0.52, y, "$z_t$", ha="left", va="center", fontsize=11.5,
                fontweight="bold", color=TEXT_DARK, zorder=6)
    elif kind == "latent_pred":
        latent_bars(ax, X_PAYLOAD - 0.42, y, w=0.95, seed=2)
        clock_icon(ax, X_PAYLOAD + 0.24, y)
        ax.text(X_PAYLOAD + 0.52, y, "$\\hat{z}_{t+1}$", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=TEXT_DARK, zorder=6)
    elif kind == "patch":
        frame_swatch(ax, X_PAYLOAD - 0.3, y, spec["frame"])
        ax.text(X_PAYLOAD + 0.14, y, "64\u00d764\u00d73", ha="left", va="center",
                fontsize=9.4, color=TEXT_DARK, zorder=6)
    elif kind == "bit":
        bit_glyph(ax, X_PAYLOAD - 0.25, y)

    ax.text(X_WIRE, y + 0.15, spec["wire"], ha="left", va="center", fontsize=9.2,
            fontweight="bold", color=TEXT_DARK if spec["wire"] == "66-d" else CEIL_BLUE,
            family="monospace", zorder=6)
    if spec.get("tag"):
        ax.text(X_WIRE, y - 0.2, spec["tag"], ha="left", va="center", fontsize=7.5,
                color=NOTE_GRAY, style="italic", zorder=6)

    result_marker(ax, y, vals, accent)


def draw_ladder_panel(ax, results, frame_path):
    row_y = {r["n"]: 0.7 + (r["n"] - 1) * 1.0 for r in ROWS}
    for r in ROWS:
        spec = dict(r)
        if spec["kind"] == "patch":
            spec["frame"] = frame_path
        draw_row(ax, row_y[r["n"]], spec, results[r["cond"]])

    # the oracle is not a rung: it asks whether the optimizer works, not
    # whether the representation carries the bit
    rule_y = 0.0
    ax.plot([X_BADGE - 0.35, BAR_X0 + BAR_W + 1.05], [rule_y, rule_y],
            color="#c7cbd1", lw=1.0, ls=(0, (5, 3)), zorder=3)
    ax.text(X_NAME, rule_y - 0.26,
            "not a rung \u2014 asks whether the optimizer works, not whether the representation carries the bit",
            ha="left", va="center", fontsize=8.3, color=NOTE_GRAY, style="italic")
    draw_row(ax, rule_y - 0.78, ORACLE, results["oracle"], dashed=True)

    # information-content axis, left of the badges
    ax.add_patch(FancyArrowPatch((X_ARROW, row_y[1] - 0.3), (X_ARROW, row_y[6] + 0.42),
                                 arrowstyle="-|>", mutation_scale=13, color=TEXT_DARK,
                                 linewidth=1.4, zorder=5))
    ax.text(X_ARROW - 0.3, (row_y[1] + row_y[6]) / 2, "information content",
            ha="center", va="center", fontsize=9.8, fontweight="bold",
            color=TEXT_DARK, rotation=90, zorder=6)

    ax.text(BAR_X0 + BAR_W / 2 + 0.3, row_y[6] + 0.78, "route-optimality",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color=TEXT_DARK)
    ax.text(BAR_X0 + BAR_W / 2 + 0.3, row_y[6] + 0.5, "dashed tick = chance",
            ha="center", va="center", fontsize=7.6, color=NOTE_GRAY, style="italic")

    ax.set_xlim(-0.45, BAR_X0 + BAR_W + 1.25)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="spike/out/hires/occl_scout_rgb_s2.png",
                    help="scout RGB frame used as the row-6 payload swatch")
    ap.add_argument("--out", default="plots/diagrams/fig_tier2_conditions.png")
    args = ap.parse_args()

    results = load_results()
    missing = [r["cond"] for r in ROWS if r["cond"] not in results]
    if missing:
        raise SystemExit(f"no runs found for {missing}")

    fig = plt.figure(figsize=(14.2, 6.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.55, 13.15], wspace=0.01)
    draw_channel_panel(fig.add_subplot(gs[0, 0]))
    draw_ladder_panel(fig.add_subplot(gs[0, 1]), results, args.frame)

    fig.suptitle("One channel, six message contents \u2014 now every one of them raced",
                 fontsize=17.0, fontweight="bold", y=0.985)
    fig.text(0.5, 0.933,
             "identical scout \u2192 route-head channel across all six conditions \u2014 only the payload changes; "
             "6,000 episodes per seed, frozen executor",
             ha="center", va="center", fontsize=10.5, color=NOTE_GRAY, style="italic")
    raw = results["raw_obs"]
    fig.text(0.5, 0.022,
             "* raw_obs seeds: " + " / ".join(f"{v:.2f}" for v in sorted(raw, reverse=True))
             + " \u2014 one seed's decision entropy collapsed to ~$10^{-6}$ a third of the way in and froze at chance. "
               "The 186\u00d7 wider wire is the less reliable one to learn from.",
             ha="center", va="center", fontsize=8.2, color=NOTE_GRAY, style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.045)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor="white")
    print(f"wrote {args.out}")
    for c, v in results.items():
        print(f"  {c:10s} {len(v)} seeds  mean {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
