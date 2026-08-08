"""Figure 1 — Tier 1 vs Tier 2: same information structure, different substrate.

Left panel: the Tier 1 gridworld chokepoint (literal generate_chokepoint_map
output) with the navigator's REAL shadowcasting visibility (envs/fov
compute_visible, radius 7) — occluded cells hatched exactly as in Tier 1's
Figure 6b, hazard sitting inside the occluded region.

Right panel: the Tier 2 scene from the navigator's corridor-choice vantage,
rendered by the occlusion-gate probe camera at 512x512 (the training input is
the same camera at 64x64). The hazard slab is metres ahead behind the
staggered baffles: 0 hazard pixels of 262,144 — the same fact as the left
panel's hatching, established by a renderer instead of an algorithm.

    python rl/plot_fig1.py --out plots/fig1_substrate.png
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

from envs.constants import HAZARD, WALL  # noqa: E402
from envs.fov import compute_visible  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402

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
FOV_RADIUS = 7  # Tier 1's extract_patch default


def draw_tier1(ax, seed: int):
    spec = generate_chokepoint_map(np.random.default_rng(seed))
    grid = spec.grid
    nav, scout = spec.agent_starts
    visible = compute_visible(grid, nav, FOV_RADIUS)
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            val = grid[r, c]
            base = WALL_C if val == WALL else HAZARD_C if val == HAZARD else FLOOR_C
            ax.add_patch(mpatches.Rectangle(
                (c, h - 1 - r), 1, 1, facecolor=base,
                edgecolor=GRID_LINE, linewidth=0.4, zorder=1,
            ))
            if not visible[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, h - 1 - r), 1, 1, facecolor=OCCLUDED_C,
                    edgecolor=GRID_LINE, linewidth=0.4, alpha=0.55,
                    hatch="////", zorder=2,
                ))

    for (r, c) in spec.goals:
        ax.scatter([c + 0.5], [h - 1 - r + 0.5], marker="*", s=150,
                   color=GOAL_C, edgecolor="black", linewidth=0.6, zorder=6)
    for (r, c), color, label in ((nav, EGO_C, "N"), (scout, MATE_C, "S")):
        cx, cy = c + 0.5, h - 1 - r + 0.5
        ax.add_patch(Circle((cx, cy), 0.42, facecolor=color,
                            edgecolor="white", linewidth=1.4, zorder=5))
        ax.text(cx, cy - 0.05, label, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=6)

    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tier 1 — gridworld chokepoint\nshadowcasting FOV (radius 7), literal algorithm output",
                 fontsize=10.5, fontweight="bold", pad=8)


def draw_tier2(ax, frame_path: str):
    img = plt.imread(frame_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Tier 2 — the same choice point, on pixels\nnavigator's probe camera; hazard slab behind the baffles",
                 fontsize=10.5, fontweight="bold", pad=8)
    ax.annotate("slab is behind this baffle:\n0 / 262,144 hazard pixels",
                xy=(0.47, 0.50), xycoords="axes fraction",
                xytext=(0.05, 0.10), textcoords="axes fraction",
                fontsize=9, color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame", type=str,
                    default="spike/out/hires/occl_navigator_mouth_bottom_rgb.png")
    ap.add_argument("--out", type=str, default="plots/fig1_substrate.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    draw_tier1(axes[0], args.seed)
    draw_tier2(axes[1], args.frame)

    legend_handles = [
        mpatches.Patch(facecolor=FLOOR_C, edgecolor=GRID_LINE, label="free cell"),
        mpatches.Patch(facecolor=WALL_C, edgecolor=GRID_LINE, label="wall"),
        mpatches.Patch(facecolor=HAZARD_C, edgecolor=GRID_LINE, label="hazard slab"),
        mpatches.Patch(facecolor=OCCLUDED_C, edgecolor=GRID_LINE, alpha=0.55,
                       hatch="////", label="occluded from navigator"),
        mpatches.Circle((0, 0), radius=0.3, facecolor=EGO_C, edgecolor="white",
                        label="N — navigator"),
        mpatches.Circle((0, 0), radius=0.3, facecolor=MATE_C, edgecolor="white",
                        label="S — scout"),
    ]
    fig.suptitle("Same information structure, different substrate",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.085,
             "left: Tier 1's tested occlusion algorithm (map seed 0) hides the hazard from the navigator; the scout sees it and only a message can carry it.\n"
             "right: the same seed-0 map extruded in Isaac Sim — ray-traced light and staggered baffles re-establish the identical fact, certified by the occlusion gate.",
             ha="center", va="center", fontsize=8.2, color=NOTE_GRAY, style="italic")
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8.3, bbox_to_anchor=(0.5, 0.008))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.17, wspace=0.06)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
