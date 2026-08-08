"""Figure — the race v8 pipeline, drawn as a closed loop.

Tier 1's box-and-arrow language: frozen modules green, trainable blue, action
output amber, and (new here) the learning signal in red.

The design point: a pipeline diagram that makes a training claim should draw
the training signal, not assert it in a caption. The whole result is that a
single scalar episode return — which names nothing, supervises nothing, and
arrives only at the end — recruits a frozen task-agnostic latent. So the
return is an arrow, the world that produces it is a node, and the two paths
that do NOT exist (head has no ego input, executor has no message input) are
stated where they belong, beneath the module they constrain.

Every number is checked against the code: RouteHead(66) has 4,483 parameters,
ROUTE_DIM = 2, DECIDE_STEP = 2, and the executor is built with
broadcast_dim=0 (rl/train_race_route.py).

    python rl/plot_fig_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

TEXT_DARK = "#20222a"
NOTE_GRAY = "#6a707a"

FROZEN_FACE, FROZEN_EDGE, FROZEN_TEXT = "#e8f3ec", "#2f7d4f", "#1f6b41"
TRAIN_FACE, TRAIN_EDGE, TRAIN_TEXT = "#e4edf9", "#2f5f8f", "#26527d"
ACTION_FACE, ACTION_EDGE, ACTION_TEXT = "#fdf0dc", "#c9821f", "#9c6413"
PLAIN_FACE, PLAIN_EDGE = "#ffffff", "#9aa1ab"
PANEL_FACE = "#f4f5f7"
WORLD_FACE, WORLD_EDGE = "#f1f3f6", "#7d8794"
GRAD_RED = "#c1292e"
HAZARD_C = "#C44E52"


def box(ax, cx, cy, w, h, lines, *, face, edge, lw=1.5, dashed=False, zorder=5):
    """lines = [(text, fontsize, weight, style, color), ...] stacked and centered."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.9",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                linestyle=(0, (4, 2)) if dashed else "solid",
                                zorder=zorder))
    gaps = [fs for _, fs, _, _, _ in lines]
    total = sum(g * 0.115 for g in gaps) + 0.55 * (len(lines) - 1)
    y = cy + total / 2
    for text, fs, weight, style, color in lines:
        y -= fs * 0.115 / 2
        ax.text(cx, y, text, ha="center", va="center", fontsize=fs,
                fontweight=weight, style=style, color=color, zorder=zorder + 1)
        y -= fs * 0.115 / 2 + 0.55


def arrow(ax, p0, p1, *, color=TEXT_DARK, lw=1.6, dashed=False, mut=13, zorder=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                                 color=color, linewidth=lw,
                                 linestyle=(0, (4.5, 2.5)) if dashed else "solid",
                                 zorder=zorder))


def panel(ax, x0, x1, y0, y1, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=1.1",
                                facecolor=PANEL_FACE, edgecolor="#dfe2e7",
                                linewidth=1.0, zorder=1))
    ax.text((x0 + x1) / 2, y1 - 2.1, title, ha="center", va="center",
            fontsize=10.4, fontweight="bold", style="italic", color=color, zorder=3)


def draw(ax, overhead):
    # ---- agent panels -----------------------------------------------------
    panel(ax, 2, 24, 34, 58, "SCOUT \u2014 static beacon", GRAD_RED)
    panel(ax, 70, 100, 16, 58, "NAVIGATOR \u2014 the learner's body", TRAIN_EDGE)

    SX, NX = 13, 85
    y_cam, y_enc, y_z = 51.5, 46.0, 38.5

    for cx, w in ((SX, 19), (NX, 26)):
        box(ax, cx, y_cam, w, 4.6,
            [("camera \u00b7 64\u00d764 RGB", 9.4, "normal", "normal", TEXT_DARK)],
            face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)
        box(ax, cx, y_enc, w, 6.2,
            [("Encoder E", 11.2, "bold", "normal", FROZEN_TEXT),
             ("JEPA \u2014 frozen", 8.4, "normal", "italic", FROZEN_TEXT)],
            face=FROZEN_FACE, edge=FROZEN_EDGE)
        arrow(ax, (cx, y_cam - 2.3), (cx, y_enc + 3.1))
        arrow(ax, (cx, y_enc - 3.1), (cx, y_z + 2.2))

    box(ax, SX, y_z, 19, 4.4,
        [("$z_t$   64-D", 10.2, "bold", "normal", TEXT_DARK)],
        face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)
    box(ax, NX, y_z, 26, 4.4,
        [("$z_{ego}$   64-D", 10.2, "bold", "normal", TEXT_DARK)],
        face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)

    # one encoder, used twice — no alignment layer anywhere
    ax.plot([23.5, 71.5], [y_enc, y_enc], color=FROZEN_EDGE, lw=1.4,
            ls=(0, (5, 3)), zorder=2)
    ax.text(47.5, y_enc + 3.4, "shared weights \u2014 one encoder, used twice",
            ha="center", va="center", fontsize=9.4, fontweight="bold",
            color=FROZEN_TEXT)
    ax.text(47.5, y_enc + 1.3, "no cross-agent alignment layer, no gradient either side",
            ha="center", va="center", fontsize=8.2, style="italic", color=FROZEN_TEXT)

    # ---- the wire ---------------------------------------------------------
    box(ax, 31, y_z, 8.4, 8.6,
        [("the wire", 9.8, "bold", "normal", TEXT_DARK),
         ("66 floats", 9.0, "bold", "normal", TRAIN_TEXT),
         ("2 anchor", 7.4, "normal", "italic", NOTE_GRAY),
         ("+ 64 content", 7.4, "normal", "italic", NOTE_GRAY)],
        face="#eef2f8", edge=TRAIN_EDGE, lw=1.3)
    arrow(ax, (22.6, y_z), (26.6, y_z))

    # ---- the route head ---------------------------------------------------
    box(ax, 51, 36, 24, 14.5,
        [("ROUTE HEAD", 12.6, "bold", "normal", TRAIN_TEXT),
         ("the only trainable module", 8.8, "normal", "italic", TRAIN_TEXT),
         ("message \u2192 2 logits + value", 9.2, "normal", "normal", TEXT_DARK),
         ("4,483 parameters", 10.4, "bold", "normal", TRAIN_TEXT),
         ("one categorical decision per episode,", 7.8, "normal", "italic", NOTE_GRAY),
         ("committed at step 2", 7.8, "normal", "italic", NOTE_GRAY)],
        face=TRAIN_FACE, edge=TRAIN_EDGE, lw=2.0)
    arrow(ax, (35.3, y_z), (38.9, 37.5))

    # ---- executor and action ---------------------------------------------
    box(ax, NX, 30, 26, 8.4,
        [("EXECUTOR", 11.6, "bold", "normal", FROZEN_TEXT),
         ("AttentionReceiver \u2014 frozen", 8.4, "normal", "italic", FROZEN_TEXT),
         ("(own rgb, route) \u2192 action", 8.8, "normal", "normal", TEXT_DARK),
         ("obedience 1.000", 9.0, "bold", "normal", FROZEN_TEXT)],
        face=FROZEN_FACE, edge=FROZEN_EDGE)
    arrow(ax, (NX, y_z - 2.2), (NX, 34.4))
    arrow(ax, (63.2, 33.5), (71.6, 31.4), color=TRAIN_EDGE)
    ax.text(67.4, 35.2, "route", ha="center", va="center", fontsize=8.4,
            fontweight="bold", color=TRAIN_EDGE)

    box(ax, NX, 21, 26, 4.8,
        [("cmd_vel  ($v_x$, $v_y$, $\\omega_z$)", 10.0, "bold", "normal", ACTION_TEXT)],
        face=ACTION_FACE, edge=ACTION_EDGE)
    arrow(ax, (NX, 25.7), (NX, 23.5), color=ACTION_EDGE)

    # ---- the world, and the only gradient in the system -------------------
    box(ax, NX, 8.6, 26, 10.6, [], face=WORLD_FACE, edge=WORLD_EDGE, lw=1.3)
    img = plt.imread(overhead)
    ax.imshow(img, extent=[74.2, 82.2, 4.6, 12.6], aspect="auto", zorder=6)
    ax.add_patch(Rectangle((74.2, 4.6), 8.0, 8.0, facecolor="none",
                           edgecolor=WORLD_EDGE, linewidth=1.0, zorder=7))
    ax.text(89.6, 11.0, "THE WORLD", ha="center", va="center", fontsize=10.6,
            fontweight="bold", color=TEXT_DARK, zorder=7)
    ax.text(89.6, 8.9, "the hazard prices the mistake", ha="center", va="center",
            fontsize=8.0, style="italic", color=NOTE_GRAY, zorder=7)
    ax.text(89.6, 6.6, "no instructor, no label", ha="center", va="center",
            fontsize=8.0, style="italic", color=HAZARD_C, zorder=7)
    arrow(ax, (NX, 18.5), (NX, 14.1), color=ACTION_EDGE)

    # the return: world -> left -> up into the head, and nowhere else
    ax.plot([71.8, 51], [8.6, 8.6], color=GRAD_RED, lw=1.7, ls=(0, (4.5, 2.5)), zorder=6)
    arrow(ax, (51, 8.6), (51, 28.4), color=GRAD_RED, lw=1.7, dashed=True)
    ax.text(61.4, 10.4, "episode return \u2014 one scalar",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=GRAD_RED)
    ax.text(61.4, 8.6 - 1.9, "it names nothing", ha="center", va="center",
            fontsize=7.8, style="italic", color=GRAD_RED)
    ax.text(52.6, 19.5, "the only gradient\nin the system", ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=GRAD_RED, linespacing=1.3)

    # ---- the two paths that do not exist ----------------------------------
    ax.add_patch(FancyBboxPatch((2, 14.4), 39, 11.6,
                                boxstyle="round,pad=0.0,rounding_size=0.9",
                                facecolor="#fdf0f0", edgecolor=GRAD_RED,
                                linewidth=1.3, zorder=5))
    ax.text(21.5, 23.6, "Message-dependence is architectural, not inferred",
            ha="center", va="center", fontsize=9.4, fontweight="bold",
            color=GRAD_RED, zorder=6)
    for i, line in enumerate((
            "the head has no ego input \u2014 it never sees the navigator's camera",
            "the executor has no message input \u2014 built with broadcast_dim = 0",
            "so the route bit can only have come off the wire")):
        ax.text(4.4, 20.6 - i * 2.1, "\u2022  " + line, ha="left", va="center",
                fontsize=8.2, color=TEXT_DARK, zorder=6)

    ax.set_xlim(0, 102)
    ax.set_ylim(2, 60)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", default="spike/out/hires/overhead_s2.png")
    ap.add_argument("--out", default="plots/diagrams/fig_tier2_pipeline.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13.6, 8.4))
    draw(ax, args.overhead)

    fig.suptitle("The race v8 pipeline \u2014 4,483 trainable parameters, and a scalar that names nothing",
                 fontsize=15.0, fontweight="bold", y=0.975)
    fig.text(0.5, 0.932,
             "encode \u2192 broadcast \u2192 decide \u2192 execute \u2192 the world prices the mistake",
             ha="center", va="center", fontsize=10.2, color=NOTE_GRAY, style="italic")

    handles = [
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0", facecolor=FROZEN_FACE,
                       edgecolor=FROZEN_EDGE, label="frozen (no gradient)"),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0", facecolor=TRAIN_FACE,
                       edgecolor=TRAIN_EDGE, label="trainable"),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0", facecolor=ACTION_FACE,
                       edgecolor=ACTION_EDGE, label="action output"),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0", facecolor=WORLD_FACE,
                       edgecolor=WORLD_EDGE, label="the world"),
        plt.Line2D([0], [0], color=FROZEN_EDGE, lw=1.5, ls=(0, (5, 3)),
                   label="shared weights"),
        plt.Line2D([0], [0], color=GRAD_RED, lw=1.7, ls=(0, (4.5, 2.5)),
                   label="learning signal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9.0, bbox_to_anchor=(0.5, 0.012))

    fig.subplots_adjust(left=0.005, right=0.995, top=0.905, bottom=0.065)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
