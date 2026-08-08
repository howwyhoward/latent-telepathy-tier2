"""Figure 1 — same information structure, different substrate.

Rows are the two substrates, columns are three viewpoints, so every column
holds the viewpoint fixed and varies only the substrate — which is the actual
claim. Mirrors Tier 1's Figure 6b grammar (world on one axis, what an agent
sees on the other).

              | the world      | corridor mouth       | scout's post
  ------------|----------------|----------------------|-------------------
  Tier 1      | grid + FOV     | cells the FOV admits | cells the FOV admits
  Tier 2      | overhead render| onboard camera       | onboard camera

Map seed 2 throughout: the slab sits in the scout's corridor, which is the
configuration the whole experiment turns on. Tier 1 panels come from the
literal shadowcasting algorithm (envs/fov.compute_visible, radius 7); Tier 2
panels are 512^2 renders from the occlusion gate, whose hazard-pixel counts
are quoted verbatim. The policy sees the same optics at 64x64.

The middle column is the corridor mouth rather than the navigator's spawn
because that is where the corridor is chosen — and it is the strict test: at
the mouth the navigator is already looking down the slabbed corridor.

    python rl/plot_fig1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Tier 1 is the source of truth for the map (same arrangement as chokepoint/geometry.py)
sys.path.insert(0, str(Path.home() / "latent-telepathy"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.constants import HAZARD, WALL  # noqa: E402
from envs.fov import compute_visible  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402

from chokepoint.geometry import chokepoint_grid  # noqa: E402

# Tier 1 figure palette (rl/plot_figure6b_fov_demo.py)
WALL_C = "#2b2d42"
FLOOR_C = "#eef1f5"
GRID_LINE = "#c9cdd3"
OCCLUDED_C = "#9aa1ab"
EGO_C = "#1f8a70"
MATE_C = "#e07a2c"
GOAL_C = "#d4a017"
HAZARD_C = "#C44E52"
NOTE_GRAY = "#6a707a"
TEXT_DARK = "#20222a"

FOV_RADIUS = 7   # Tier 1's extract_patch default
PATCH = 15       # Tier 1's patch_size
MOUTH = (7, 4)   # top-corridor mouth; the Tier 2 probe camera stands here


def cell_face(val):
    if val == WALL:
        return WALL_C
    if val == HAZARD:
        return HAZARD_C
    return FLOOR_C


def hazard_readout(ax, text, *, dark_bg):
    ax.text(0.5, 0.035, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.2, fontweight="bold",
            color="white" if dark_bg else TEXT_DARK,
            bbox=dict(facecolor="black" if dark_bg else "#f1f3f6", alpha=0.6 if dark_bg else 1.0,
                      edgecolor="none" if dark_bg else "#c9cdd3",
                      boxstyle="round,pad=0.34"),
            zorder=10)


def draw_world(ax, grid, spec, title):
    """Tier 1's world: true cells, plus the two vantages' shadowcasting FOV."""
    h, w = grid.shape
    vis_mouth = compute_visible(grid, MOUTH, FOV_RADIUS)
    vis_scout = compute_visible(grid, spec.agent_starts[1], FOV_RADIUS)

    for r in range(h):
        for c in range(w):
            ax.add_patch(mpatches.Rectangle((c, h - 1 - r), 1, 1,
                                            facecolor=cell_face(grid[r, c]),
                                            edgecolor=GRID_LINE, linewidth=0.4, zorder=1))
            for vis, color, alpha in ((vis_mouth, EGO_C, 0.30), (vis_scout, MATE_C, 0.34)):
                if vis[r, c]:
                    ax.add_patch(mpatches.Rectangle((c, h - 1 - r), 1, 1, facecolor=color,
                                                    edgecolor="none", alpha=alpha, zorder=2))

    for (r, c) in spec.goals:
        ax.scatter([c + 0.5], [h - 1 - r + 0.5], marker="*", s=125, color=GOAL_C,
                   edgecolor="black", linewidth=0.6, zorder=6)

    nav, scout = spec.agent_starts
    marker = ((nav, "#5b6472", "N", False), (MOUTH, EGO_C, "M", True),
              (scout, MATE_C, "S", True))
    for (r, c), color, label, filled in marker:
        cx, cy = c + 0.5, h - 1 - r + 0.5
        ax.add_patch(Circle((cx, cy), 0.46, facecolor=color if filled else "white",
                            edgecolor="white" if filled else color, linewidth=1.5, zorder=5))
        ax.text(cx, cy - 0.05, label, ha="center", va="center",
                color="white" if filled else color, fontsize=9.5,
                fontweight="bold", zorder=6)

    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.8, fontweight="bold", pad=7, linespacing=1.35)


def draw_patch(ax, grid, origin, title, *, ego_color, ego_label):
    """The 15x15 egocentric footprint: cells the shadowcaster admits carry
    their true content, everything else (occluded or out of bounds) is the
    single UNKNOWN class — from the agent's seat those are the same thing."""
    h, w = grid.shape
    vis = compute_visible(grid, origin, FOV_RADIUS)
    half = PATCH // 2
    r0, c0 = origin[0] - half, origin[1] - half
    seen_hazard = 0

    for pr in range(PATCH):
        for pc in range(PATCH):
            r, c = r0 + pr, c0 + pc
            inside = 0 <= r < h and 0 <= c < w
            known = inside and vis[r, c]
            face = cell_face(grid[r, c]) if known else OCCLUDED_C
            ax.add_patch(mpatches.Rectangle((pc, PATCH - 1 - pr), 1, 1, facecolor=face,
                                            edgecolor=GRID_LINE, linewidth=0.4,
                                            alpha=1.0 if known else 0.55,
                                            hatch=None if known else "////", zorder=1))
            if known and grid[r, c] == HAZARD:
                seen_hazard += 1

    cx = cy = half + 0.5
    ax.add_patch(Circle((cx, cy), 0.46, facecolor=ego_color, edgecolor="white",
                        linewidth=1.5, zorder=5))
    ax.text(cx, cy - 0.05, ego_label, ha="center", va="center", color="white",
            fontsize=9.5, fontweight="bold", zorder=6)

    ax.set_xlim(-0.2, PATCH + 0.2)
    ax.set_ylim(-0.2, PATCH + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.8, fontweight="bold", pad=7, linespacing=1.35)
    return seen_hazard


def draw_image(ax, path, title):
    ax.imshow(plt.imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.8, fontweight="bold", pad=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--hires", default="spike/out/hires")
    ap.add_argument("--out", default="plots/fig1_substrate.png")
    args = ap.parse_args()

    grid = chokepoint_grid(args.seed)          # the grid Tier 2 extrudes
    spec = generate_chokepoint_map(np.random.default_rng(args.seed))
    n_hazard = int((grid == HAZARD).sum())
    hires = Path(args.hires)

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.7))
    heads = ["the world\noverhead \u2014 nobody's viewpoint",
             "at the corridor mouth\nwhere the route is chosen",
             "the scout's post\nthe vantage that holds the bit"]

    draw_world(axes[0, 0], grid, spec, heads[0])
    seen_mouth = draw_patch(axes[0, 1], grid, MOUTH, heads[1],
                            ego_color=EGO_C, ego_label="M")
    seen_scout = draw_patch(axes[0, 2], grid, spec.agent_starts[1], heads[2],
                            ego_color=MATE_C, ego_label="S")
    hazard_readout(axes[0, 1], f"hazard cells in view:  {seen_mouth} / {n_hazard}", dark_bg=False)
    hazard_readout(axes[0, 2], f"hazard cells in view:  {seen_scout} / {n_hazard}", dark_bg=False)

    draw_image(axes[1, 0], hires / f"overhead_s{args.seed}.png", None)
    draw_image(axes[1, 1], hires / f"occl_navigator_mouth_top_rgb_s{args.seed}.png", None)
    draw_image(axes[1, 2], hires / f"occl_scout_rgb_s{args.seed}.png", None)
    hazard_readout(axes[1, 1], "hazard pixels:  0 / 262,144", dark_bg=True)
    hazard_readout(axes[1, 2], "hazard pixels:  89,475 / 262,144", dark_bg=True)
    axes[1, 1].annotate("slab is behind these\nstaggered baffles",
                        xy=(0.45, 0.50), xycoords="axes fraction",
                        xytext=(0.04, 0.20), textcoords="axes fraction",
                        fontsize=8.6, color="white",
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.1))

    fig.suptitle("Same information structure, different substrate",
                 fontsize=15.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.952,
             "rows are substrates, columns are viewpoints \u2014 map seed 2 in both tiers, "
             "with the hazard slab in the scout's corridor",
             ha="center", va="center", fontsize=9.8, color=NOTE_GRAY, style="italic")

    for y, label, sub in ((0.715, "Tier 1", "discrete cells\nshadowcasting FOV"),
                          (0.300, "Tier 2", "rendered pixels\nray-traced light")):
        fig.text(0.016, y, label, ha="center", va="center", fontsize=12.5,
                 fontweight="bold", color=TEXT_DARK, rotation=90)
        fig.text(0.046, y, sub, ha="center", va="center", fontsize=8.6,
                 color=NOTE_GRAY, style="italic", rotation=90, linespacing=1.3)

    legend_handles = [
        mpatches.Patch(facecolor=FLOOR_C, edgecolor=GRID_LINE, label="free cell"),
        mpatches.Patch(facecolor=WALL_C, edgecolor=GRID_LINE, label="wall"),
        mpatches.Patch(facecolor=HAZARD_C, edgecolor=GRID_LINE, label="hazard slab"),
        mpatches.Patch(facecolor=OCCLUDED_C, edgecolor=GRID_LINE, alpha=0.55,
                       hatch="////", label="occluded / out of bounds"),
        mpatches.Patch(facecolor=EGO_C, alpha=0.30, label="FOV from the mouth (M)"),
        mpatches.Patch(facecolor=MATE_C, alpha=0.34, label="scout FOV (S)"),
        mpatches.Circle((0, 0), radius=0.3, facecolor="white", edgecolor="#5b6472",
                        label="N \u2014 navigator spawn"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, 0.043))
    fig.text(0.5, 0.014,
             "Tier 1 hides the slab with a tested shadowcasting algorithm on discrete cells; Tier 2 re-establishes the identical fact on the same extruded grid "
             "with ray-traced light and staggered baffles,\ncertified by the occlusion gate (re-run at 512\u00b2 for this figure; the policy input is the same camera at 64\u00b2). "
             "The Tier 2 overhead is nobody's viewpoint \u2014 it shows only that the floor plans agree.",
             ha="center", va="bottom", fontsize=8.0, color=NOTE_GRAY, style="italic",
             linespacing=1.45)

    fig.subplots_adjust(left=0.062, right=0.995, top=0.895, bottom=0.105,
                        wspace=0.03, hspace=0.07)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  tier1 hazard cells in view: mouth {seen_mouth}/{n_hazard}, "
          f"scout {seen_scout}/{n_hazard}")


if __name__ == "__main__":
    main()
