"""Figure 3 — the occlusion gate, literal renderer output.

Tier 2's analog of Tier 1's shadowcasting figure. Tier 1 could prove occlusion
by construction (recursive symmetric shadowcasting is exact); a camera in a 3D
corridor cannot, so occlusion here is a measurement on the real renderer, at
the encoder's own resolution.

Every panel is a committed 64x64 frame from spike/verify_occlusion.py, upscaled
nearest-neighbour. The blockiness is the point: this is what the encoder sees.

Frames and counts come from three gate invocations (see runs/gate/*.json):

    python spike/verify_occlusion.py --seed 2 --tag _s2 \\
        --json_out runs/gate/occl_s2.json                     # slab TOP
    python spike/verify_occlusion.py --seed 0 --tag _s0 \\
        --json_out runs/gate/occl_s0.json                     # slab BOTTOM
    python spike/verify_occlusion.py --seed 2 --no_baffles \\
        --tag _s2_nobaffle --json_out runs/gate/occl_s2_nobaffle.json

Usage:
    python rl/plot_fig3_occlusion.py --out-dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DPI = 130
RED = "#C44E52"
GREEN = "#55A868"
DARK = "#333333"
UPSCALE = 6  # nearest-neighbour, for print legibility only

# (view, tag, column title, what the gate demands)
PANELS = [
    ("navigator", "_s2", "navigator\nat the start",
     "0 hazard px", "must be 0"),
    ("scout", "_s2", "scout at its post\nslab in ITS corridor",
     "hazard px > 0", "must be > 0"),
    ("scout", "_s0", "scout at its post\nslab in the OTHER corridor",
     "0 hazard px", "must be 0"),
    ("navigator_mouth_top", "_s2", "choice point, top mouth\nWITH baffles",
     "0 hazard px", "must be 0"),
    ("navigator_mouth_top", "_s2_nobaffle", "choice point, top mouth\nNO baffles",
     "leak", "must be 0"),
]


def load(out_dir: Path, view: str, tag: str):
    rgb = plt.imread(out_dir / f"occl_{view}{tag}_rgb.png")
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)
    mask = np.load(out_dir / f"occl_{view}{tag}_hazmask.npy")
    return rgb[..., :3], mask


def upscale(a, k=UPSCALE):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=str, default="spike/out")
    ap.add_argument("--gate-dir", type=str, default="runs/gate")
    ap.add_argument("--out-dir", type=str, default="plots")
    args = ap.parse_args()

    frames = Path(args.frames_dir)
    gates = {p.stem: json.loads(p.read_text())
             for p in Path(args.gate_dir).glob("occl_*.json")}

    fig, axes = plt.subplots(2, len(PANELS), figsize=(14.6, 7.9))

    for j, (view, tag, title, _, demand) in enumerate(PANELS):
        rgb, mask = load(frames, view, tag)
        count = int(mask.sum())
        total = mask.size
        gate = gates[f"occl{tag}"]
        # the gate's own verdict for this specific probe
        if view == "navigator":
            ok = count == 0
        elif view == "scout":
            ok = count > 0 if gate["slab_side"] == "TOP" else count == 0
        else:
            ok = count == 0

        ax = axes[0, j]
        ax.imshow(upscale(rgb))
        ax.set_title(title, fontsize=10.5, color=DARK, pad=7)

        # row 2: exactly the pixels the gate counts, on a dimmed frame
        ax2 = axes[1, j]
        gray = rgb.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.45
        painted = gray.astype(np.uint8)
        painted[mask] = np.array([255, 40, 60], dtype=np.uint8)
        ax2.imshow(upscale(painted))
        colour = GREEN if ok else RED
        verdict = "PASS" if ok else "FAIL"

        if 0 < count < 200:  # a leak this small needs pointing at, in both rows
            ys, xs = np.nonzero(mask)
            cy, cx = (ys.mean() + 0.5) * UPSCALE, (xs.mean() + 0.5) * UPSCALE
            for a in (ax, ax2):
                a.add_patch(mpatches.Circle((cx, cy), 40, fill=False, ec=RED, lw=2.4))
                a.annotate(f"{count} px leak", xy=(cx, cy + 40),
                           xytext=(0.06, 0.82), textcoords="axes fraction",
                           fontsize=12, color=RED, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                     ec=RED, lw=1.2, alpha=0.95),
                           arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))

        for a in (ax, ax2):
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(colour)
                s.set_linewidth(2.6)
        ax2.text(0.5, -0.055,
                 f"{count} / {total} hazard px\n{verdict}   ({demand})",
                 transform=ax2.transAxes, ha="center", va="top", fontsize=11,
                 color=colour, fontweight="bold", linespacing=1.4)

    fig.suptitle("Figure 3 — the occlusion gate: certified on the renderer, not asserted "
                 "by an algorithm", fontsize=14.5, y=0.978)
    fig.text(0.5, 0.018,
             "Three criteria, pre-registered before the scene was built: the navigator must see zero hazard "
             "pixels; the scout must see the slab if and only if it is in the scout's own corridor (absence is "
             "signal);\nand both corridor-mouth probes — the choice points — must see zero. The rightmost column "
             "rebuilds the straight corridor the baffles replaced: 10 pixels leak to the choice point, enough "
             "for the\nnavigator to solve the task from its own camera and silently void the experiment. The "
             "baffles were designed against that measurement, and the gate is rerun after any scene edit. The "
             "gate and\nthe RL environment import one scene builder, so certified geometry and trained geometry "
             "cannot diverge. Frames are the encoder's own 64x64, upscaled nearest-neighbour; the blockiness "
             "is the point.",
             ha="center", va="bottom", fontsize=9.0, color="#555555", linespacing=1.55)
    # explicit layout: tight_layout fights equal-aspect image axes
    fig.subplots_adjust(left=0.055, right=0.995, top=0.885, bottom=0.175,
                        wspace=0.06, hspace=0.20)
    fig.text(0.020, 0.745, "RGB", rotation=90, ha="center", va="center",
             fontsize=12.5, color=DARK)
    fig.text(0.020, 0.395, "hazard class,\nsegmentation mask", rotation=90,
             ha="center", va="center", fontsize=12.5, color=DARK)

    out = Path(args.out_dir) / "diagnostics" / "fig3_occlusion_gate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    for name, g in sorted(gates.items()):
        print(f"  {name}: baffles={g['baffles']} slab={g['slab_side']} "
              f"{g['hazard_pixels']} overall={'PASS' if g['gates']['overall'] else 'FAIL'}")


if __name__ == "__main__":
    main()
