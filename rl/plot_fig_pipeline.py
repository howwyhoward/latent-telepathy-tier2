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
    # The wire lives at the bottom of the scout's stack so both agent columns
    # reach down the frame; that is what closes the dead quadrant a shorter
    # sender column would leave under it.
    panel(ax, 2, 26, 25, 58, "SCOUT \u2014 static beacon", GRAD_RED)
    panel(ax, 71, 101, 17, 58, "NAVIGATOR \u2014 the learner's body", TRAIN_EDGE)

    SX, NX = 14, 86
    y_cam, y_enc, y_z = 52.0, 46.0, 38.5

    for cx, w in ((SX, 20), (NX, 27)):
        box(ax, cx, y_cam, w, 5.0,
            [("camera \u00b7 64\u00d764 RGB", 10.6, "normal", "normal", TEXT_DARK)],
            face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)
        box(ax, cx, y_enc, w, 6.8,
            [("Encoder E", 12.4, "bold", "normal", FROZEN_TEXT),
             ("JEPA \u2014 frozen", 9.4, "normal", "italic", FROZEN_TEXT)],
            face=FROZEN_FACE, edge=FROZEN_EDGE)
        arrow(ax, (cx, y_cam - 2.5), (cx, y_enc + 3.4))
        arrow(ax, (cx, y_enc - 3.4), (cx, y_z + 2.4))

    box(ax, SX, y_z, 20, 4.8,
        [("$z_t$   64-D", 11.4, "bold", "normal", TEXT_DARK)],
        face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)
    box(ax, NX, y_z, 27, 4.8,
        [("$z_{ego}$   64-D", 11.4, "bold", "normal", TEXT_DARK)],
        face=PLAIN_FACE, edge=PLAIN_EDGE, lw=1.2)

    # one encoder, used twice — no alignment layer anywhere
    ax.plot([25.5, 71.5], [y_enc, y_enc], color=FROZEN_EDGE, lw=1.5,
            ls=(0, (5, 3)), zorder=2)
    ax.text(48.5, y_enc + 4.2, "shared weights \u2014 one encoder, used twice",
            ha="center", va="center", fontsize=10.8, fontweight="bold",
            color=FROZEN_TEXT)
    ax.text(48.5, y_enc + 1.8, "no cross-agent alignment layer, no gradient either side",
            ha="center", va="center", fontsize=9.4, style="italic", color=FROZEN_TEXT)

    # ---- the wire ---------------------------------------------------------
    box(ax, SX, 30.5, 20, 7.4,
        [("the wire \u00b7 66 floats", 11.4, "bold", "normal", TRAIN_TEXT),
         ("2 anchor + 64 content", 9.2, "normal", "italic", NOTE_GRAY)],
        face="#eef2f8", edge=TRAIN_EDGE, lw=1.5)
    arrow(ax, (SX, y_z - 2.4), (SX, 34.5))

    # ---- the route head ---------------------------------------------------
    box(ax, 52, 35.5, 26, 16.5,
        [("ROUTE HEAD", 14.0, "bold", "normal", TRAIN_TEXT),
         ("the only trainable module", 9.8, "normal", "italic", TRAIN_TEXT),
         ("message \u2192 2 logits + value", 10.4, "normal", "normal", TEXT_DARK),
         ("4,483 parameters", 11.8, "bold", "normal", TRAIN_TEXT),
         ("one categorical decision per episode,", 8.8, "normal", "italic", NOTE_GRAY),
         ("committed at step 2", 8.8, "normal", "italic", NOTE_GRAY)],
        face=TRAIN_FACE, edge=TRAIN_EDGE, lw=2.0)
    arrow(ax, (24.3, 30.5), (38.6, 33.4))

    # ---- executor and action ---------------------------------------------
    box(ax, NX, 29.5, 27, 9.6,
        [("EXECUTOR", 12.8, "bold", "normal", FROZEN_TEXT),
         ("AttentionReceiver \u2014 frozen", 9.4, "normal", "italic", FROZEN_TEXT),
         ("(own rgb, route) \u2192 action", 9.8, "normal", "normal", TEXT_DARK),
         ("obedience 1.000", 10.2, "bold", "normal", FROZEN_TEXT)],
        face=FROZEN_FACE, edge=FROZEN_EDGE)
    arrow(ax, (NX, y_z - 2.4), (NX, 34.5))
    arrow(ax, (65.3, 34.0), (72.2, 30.6), color=TRAIN_EDGE)
    ax.text(68.9, 34.6, "route", ha="center", va="center", fontsize=9.6,
            fontweight="bold", color=TRAIN_EDGE)

    box(ax, NX, 20.8, 27, 5.2,
        [("cmd_vel  ($v_x$, $v_y$, $\\omega_z$)", 11.2, "bold", "normal", ACTION_TEXT)],
        face=ACTION_FACE, edge=ACTION_EDGE)
    arrow(ax, (NX, 24.6), (NX, 23.6), color=ACTION_EDGE)

    # ---- the world: a full-width floor the whole pipeline stands on -------
    box(ax, 71, 10.4, 60, 11.6, [], face=WORLD_FACE, edge=WORLD_EDGE, lw=1.4)
    img = plt.imread(overhead)
    ax.imshow(img, extent=[43.4, 53.6, 5.3, 15.5], aspect="auto", zorder=6)
    ax.add_patch(Rectangle((43.4, 5.3), 10.2, 10.2, facecolor="none",
                           edgecolor=WORLD_EDGE, linewidth=1.0, zorder=7))
    ax.text(78.5, 13.6, "THE WORLD", ha="center", va="center", fontsize=12.6,
            fontweight="bold", color=TEXT_DARK, zorder=7)
    ax.text(78.5, 10.7, "the hazard slab prices a wrong turn", ha="center",
            va="center", fontsize=10.0, style="italic", color=NOTE_GRAY, zorder=7)
    ax.text(78.5, 7.7, "no instructor, no label \u2014 the return is all that comes back",
            ha="center", va="center", fontsize=10.0, style="italic",
            color=HAZARD_C, zorder=7)
    arrow(ax, (NX, 18.1), (NX, 16.5), color=ACTION_EDGE)

    # the return: straight up into the head, and nowhere else. It runs in the
    # one clear lane between the note and the navigator column.
    arrow(ax, (46, 16.3), (46, 27.0), color=GRAD_RED, lw=1.9, dashed=True)
    ax.text(47.8, 23.6, "episode return \u2014 one scalar,\nthe only gradient in the system",
            ha="left", va="center", fontsize=10.4, fontweight="bold",
            color=GRAD_RED, linespacing=1.35)
    ax.text(47.8, 19.6, "it names nothing", ha="left", va="center", fontsize=9.4,
            style="italic", color=GRAD_RED)

    # ---- the two paths that do not exist ----------------------------------
    ax.add_patch(FancyBboxPatch((2, 5.0), 38, 17.0,
                                boxstyle="round,pad=0.0,rounding_size=0.9",
                                facecolor="#fdf0f0", edgecolor=GRAD_RED,
                                linewidth=1.4, zorder=5))
    ax.text(21, 19.6, "Message-dependence is architectural,\nnot inferred",
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color=GRAD_RED, zorder=6, linespacing=1.3)
    for i, line in enumerate((
            "the head has no ego input \u2014 it never",
            "     sees the navigator's camera",
            "the executor has no message input \u2014",
            "     built with broadcast_dim = 0",
            "so the route bit can only have come",
            "     off the wire")):
        bullet = "\u2022  " if i % 2 == 0 else "    "
        ax.text(4.2, 15.9 - i * 1.85, bullet + line, ha="left", va="center",
                fontsize=9.6, color=TEXT_DARK, zorder=6)

    ax.set_xlim(0.5, 102.5)
    ax.set_ylim(3.5, 59.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", default="spike/out/hires/overhead_s2.png")
    ap.add_argument("--out", default="plots/diagrams/fig_tier2_pipeline.png")
    args = ap.parse_args()

    # sized so the axes aspect matches the 102x56 content box exactly —
    # otherwise aspect="equal" letterboxes it with white bars
    fig, ax = plt.subplots(figsize=(13.4, 8.75))
    draw(ax, args.overhead)

    fig.suptitle("The race v8 pipeline \u2014 4,483 trainable parameters, and a scalar that names nothing",
                 fontsize=16.5, fontweight="bold", y=0.978)
    fig.text(0.5, 0.937,
             "encode \u2192 broadcast \u2192 decide \u2192 execute \u2192 the world prices the mistake",
             ha="center", va="center", fontsize=11.4, color=NOTE_GRAY, style="italic")

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
               fontsize=10.2, bbox_to_anchor=(0.5, 0.008))

    fig.subplots_adjust(left=0.005, right=0.995, top=0.912, bottom=0.072)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
