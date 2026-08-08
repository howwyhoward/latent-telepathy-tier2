"""Figure 1 — Tier 1 vs Tier 2: same information structure, different substrate.

Left panel: the Tier 1 chokepoint on the gridworld substrate (seed 2 = the
slab-in-scout's-corridor configuration), drawn from chokepoint_grid — i.e.
with Tier 2's single geometry edit applied (mid-map rung sealed), so it is the
literal floor plan of the scene in the other two panels. Both agents' real
shadowcasting visibility (envs/fov compute_visible, radius 7) as tinted
shading: the hazard sits inside the scout's FOV and outside the navigator's —
the asymmetry the message has to bridge.

Middle/right panels: the same seed-2 map extruded in Isaac Sim, seen through
the occlusion-gate cameras rendered at 512x512 (the policy input is the same
optics at 64x64). Middle: the scout's camera — the slab fills a third of the
frame (89,475/262,144 hazard pixels). Right: the probe camera at the slabbed
corridor's mouth, same height and optics as the navigator's onboard camera —
the staggered baffles hide the slab completely (0/262,144 hazard pixels).

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.constants import HAZARD, WALL  # noqa: E402
from envs.fov import compute_visible  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402

from chokepoint.geometry import chokepoint_grid  # noqa: E402

# Tier 1 figure palette (rl/plot_figure6b_fov_demo.py)
WALL_C = "#2b2d42"
FLOOR_C = "#eef1f5"
GRID_LINE = "#c9cdd3"
EGO_C = "#1f8a70"
MATE_C = "#e07a2c"
GOAL_C = "#d4a017"
HAZARD_C = "#C44E52"
NOTE_GRAY = "#6a707a"
FOV_RADIUS = 7  # Tier 1's extract_patch default


def draw_tier1(ax, seed: int):
    spec = generate_chokepoint_map(np.random.default_rng(seed))
    # the exact grid the Tier 2 scene extrudes: Tier 2's single geometry edit
    # seals the mid-map rung so the corridor choice is irreversible
    grid = chokepoint_grid(seed)
    nav, scout = spec.agent_starts
    vis_nav = compute_visible(grid, nav, FOV_RADIUS)
    vis_scout = compute_visible(grid, scout, FOV_RADIUS)
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            val = grid[r, c]
            base = WALL_C if val == WALL else HAZARD_C if val == HAZARD else FLOOR_C
            ax.add_patch(mpatches.Rectangle(
                (c, h - 1 - r), 1, 1, facecolor=base,
                edgecolor=GRID_LINE, linewidth=0.4, zorder=1,
            ))
            # FOV-radius shading: each agent's literal shadowcasting output
            if vis_nav[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, h - 1 - r), 1, 1, facecolor=EGO_C,
                    edgecolor="none", alpha=0.28, zorder=2,
                ))
            if vis_scout[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c, h - 1 - r), 1, 1, facecolor=MATE_C,
                    edgecolor="none", alpha=0.32, zorder=2,
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
    ax.set_title("Tier 1 map, gridworld substrate\nshadowcasting FOV (radius 7), both agents",
                 fontsize=10.5, fontweight="bold", pad=8)


def draw_frame(ax, frame_path: str, title: str, note: str, note_color: str):
    img = plt.imread(frame_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax.text(0.5, 0.045, note, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=note_color,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none",
                      boxstyle="round,pad=0.35"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2,
                    help="map seed; 2 = slab in the scout's corridor")
    ap.add_argument("--scout_frame", type=str,
                    default="spike/out/hires/occl_scout_rgb_s2.png")
    ap.add_argument("--mouth_frame", type=str,
                    default="spike/out/hires/occl_navigator_mouth_top_rgb_s2.png")
    ap.add_argument("--out", type=str, default="plots/fig1_substrate.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2))
    draw_tier1(axes[0], args.seed)
    draw_frame(
        axes[1], args.scout_frame,
        "Tier 2 — scout's onboard camera\nslab in view",
        "hazard pixels: 89,475 / 262,144", "#ff9a9a",
    )
    draw_frame(
        axes[2], args.mouth_frame,
        "Tier 2 — slabbed corridor's mouth\nnavigator optics, occlusion-gate probe",
        "hazard pixels: 0 / 262,144", "white",
    )
    axes[2].annotate("slab is behind these\nstaggered baffles",
                     xy=(0.44, 0.52), xycoords="axes fraction",
                     xytext=(0.06, 0.20), textcoords="axes fraction",
                     fontsize=9, color="white",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    legend_handles = [
        mpatches.Patch(facecolor=FLOOR_C, edgecolor=GRID_LINE, label="free cell"),
        mpatches.Patch(facecolor=WALL_C, edgecolor=GRID_LINE, label="wall"),
        mpatches.Patch(facecolor=HAZARD_C, edgecolor=GRID_LINE, label="hazard slab"),
        mpatches.Patch(facecolor=EGO_C, alpha=0.28, label="navigator FOV"),
        mpatches.Patch(facecolor=MATE_C, alpha=0.32, label="scout FOV"),
        mpatches.Circle((0, 0), radius=0.3, facecolor=EGO_C, edgecolor="white",
                        label="N — navigator"),
        mpatches.Circle((0, 0), radius=0.3, facecolor=MATE_C, edgecolor="white",
                        label="S — scout"),
    ]
    fig.suptitle("Same information structure, different substrate (map seed 2, both tiers)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.085,
             "the hazard lies inside the scout's field of view and outside the navigator's — in Tier 1 by a tested shadowcasting algorithm on discrete cells,\n"
             "in Tier 2 by ray-traced light and staggered baffles on the same extruded grid (drawn with Tier 2's one edit: the mid-map rung sealed, making the\n"
             "corridor choice irreversible), certified by the occlusion gate — here re-run at 512², the policy input is the same camera at 64².",
             ha="center", va="center", fontsize=8.2, color=NOTE_GRAY, style="italic")
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, frameon=False,
               fontsize=8.3, bbox_to_anchor=(0.5, 0.008))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.19, wspace=0.05)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
