"""Figure 2b — Phase 0 feasibility: throughput, and the semantic channel.

The decision this figure records: whether RL-on-pixels was affordable on lab
hardware at all, or whether Tier 2 had to fall back to gridworld-policy plus
pixel-encoder validation. It was affordable, so no fallback was needed.

Both panels come from one committed artifact set, regenerated 2026-08-10 so the
numbers trace to a log rather than to prose:

    for n in 1 4 16 64; do
      python spike/spike_fps_benchmark.py --num_envs $n --resolution 64
    done                                  # tee'd to runs/spike/fps_benchmark.log

The right panel doubles as the segmentation wiring check — the same check that
later becomes the occlusion gate. It also shows the collection gotcha: the
simulator's idToLabels table grows lazily, so class ids are NOT stable across
invocations and the remap must be recomputed rather than cached.

Usage:
    python rl/plot_fig2b_throughput.py --out-dir plots
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 130
RED = "#C44E52"
BLUE = "#4C72B0"
DARK = "#333333"
GREEN = "#55A868"
UPSCALE = 6

CLASS_COLORS = {  # by class NAME, never by id — see the gotcha note
    "BACKGROUND": "#f2f2f2",
    "UNLABELLED": "#cfcfcf",
    "wall": "#8c8c8c",
    "hazard": RED,
    "agent": BLUE,
}

RESULT_RE = re.compile(
    r"RESULT num_envs=(\d+) res=(\d+) sim_steps/s=([\d.]+) "
    r"env_steps/s=([\d.]+) \(([\d.]+)M env-steps/GPU-day\)")


def parse_log(path: Path):
    rows, maps = [], {}
    n_seen = None
    for line in path.read_text().splitlines():
        m = RESULT_RE.search(line)
        if m:
            rows.append(dict(num_envs=int(m.group(1)), res=int(m.group(2)),
                             sim_sps=float(m.group(3)), env_sps=float(m.group(4)),
                             gpu_day=float(m.group(5))))
            continue
        m = re.search(r"num_envs=(\d+), res=\d+", line)
        if m:
            n_seen = int(m.group(1))
        if "seg id mapping" in line and n_seen is not None:
            payload = line.split("seg id mapping:", 1)[1].strip()
            maps[n_seen] = ast.literal_eval(payload)["idToLabels"]
    if not rows:
        raise SystemExit(f"no RESULT lines in {path}")
    return sorted(rows, key=lambda r: r["num_envs"]), maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="runs/spike/fps_benchmark.log")
    ap.add_argument("--frames-dir", type=str, default="spike/out")
    ap.add_argument("--out-dir", type=str, default="plots")
    args = ap.parse_args()

    rows, maps = parse_log(Path(args.log))
    top = rows[-1]

    fig = plt.figure(figsize=(13.4, 5.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 1, 1],
                          height_ratios=[1, 0.50], wspace=0.30, hspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_seg = fig.add_subplot(gs[0, 2])
    ax_note = fig.add_subplot(gs[1, 1:])

    # -- left: throughput
    n = np.array([r["num_envs"] for r in rows], float)
    sps = np.array([r["env_sps"] for r in rows], float)
    ideal = sps[0] * n / n[0]
    ax.plot(n, ideal, ":", color="#999999", lw=1.6, label="linear in num_envs")
    ax.plot(n, sps, "o-", color=RED, lw=2.4, ms=9, label="measured")
    for r in rows:
        dy = 8 if r is rows[0] else -12
        ax.annotate(f"{r['env_sps']:.0f}", xy=(r["num_envs"], r["env_sps"]),
                    xytext=(8, dy), textcoords="offset points",
                    fontsize=9.5, color=DARK)
    ax.scatter([top["num_envs"]], [top["env_sps"]], s=240, facecolor="none",
               edgecolor=RED, lw=2.3, zorder=5)
    ax.annotate(f"{top['gpu_day']:.0f}M env-steps / GPU-day\n"
                f"at {top['num_envs']} envs, "
                f"{2 * top['num_envs']} tiled cameras",
                xy=(top["num_envs"], top["env_sps"]), xytext=(1.55, 3400),
                fontsize=10.5, color=RED,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{int(v)}" for v in n], fontsize=10)
    ax.set_xlabel("parallel environments (2 cameras each, 64x64, DLSS off)", fontsize=11)
    ax.set_ylabel("environment steps / second", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.set_title("(a) rendering cost is amortized, not paid per environment",
                 fontsize=11.5, loc="left")
    ax.text(0.97, 0.06,
            f"simulation rate falls only {rows[0]['sim_sps']:.0f} -> "
            f"{top['sim_sps']:.0f} steps/s\nacross a {int(n[-1] / n[0])}x increase in "
            "cameras, so throughput\nscales near-linearly with num_envs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=DARK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#dddddd"))

    # -- right: the semantic channel, at the configuration that was adopted
    tag = f"e{top['num_envs']}_r{top['res']}"
    rgb = np.load(Path(args.frames_dir) / f"rgb_{tag}.npy")
    seg = np.load(Path(args.frames_dir) / f"seg_{tag}.npy").squeeze()
    id_to_name = {int(k): v["class"] for k, v in maps[top["num_envs"]].items()}

    def up(a):
        return np.repeat(np.repeat(a, UPSCALE, axis=0), UPSCALE, axis=1)

    ax_rgb.imshow(up(rgb.astype(np.uint8)))
    ax_rgb.set_title("RGB", fontsize=11, color=DARK)

    painted = np.zeros((*seg.shape, 3), dtype=float)
    counts = {}
    for sid, name in id_to_name.items():
        m = seg == sid
        counts[name] = int(m.sum())
        painted[m] = matplotlib.colors.to_rgb(CLASS_COLORS.get(name, "#000000"))
    ax_seg.imshow(up(painted))
    ax_seg.set_title("semantic segmentation", fontsize=11, color=DARK)
    for a in (ax_rgb, ax_seg):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#cccccc")

    handles = [plt.Line2D([], [], marker="s", ls="", ms=9,
                          color=CLASS_COLORS.get(nm, "#000000"),
                          label=f"{nm}  {counts.get(nm, 0)} px")
               for nm in ("wall", "hazard", "agent") if nm in counts]
    ax_seg.legend(handles=handles, fontsize=8.2, loc="lower left",
                  framealpha=0.92, borderpad=0.5)

    # -- the gotcha, stated with the evidence
    ax_note.axis("off")
    orderings = "\n".join(
        f"      num_envs={k:>2}:  " + ", ".join(
            f"{i}={v['class']}" for i, v in sorted(maps[k].items(), key=lambda kv: int(kv[0])))
        for k in sorted(maps))
    ax_note.text(
        0.0, 1.0,
        "Collection gotcha, recorded because it silently mislabels the rare classes that matter most:\n"
        "the idToLabels table grows lazily, so class ids are not stable across invocations —\n"
        f"{orderings}\n"
        "The remap must therefore be recomputed from the table, never cached by id.",
        transform=ax_note.transAxes, ha="left", va="top", fontsize=8.6,
        color=DARK, linespacing=1.6, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fbf7ee", ec="#e0d6bd"))

    fig.suptitle("Figure 2b — Phase 0: the sample budget exists, and the semantic channel is wired",
                 fontsize=13.5, y=0.975)
    fig.text(0.5, 0.012,
             "RTX 6000 Ada, headless, 500 timed steps after 50 warmup, antialiasing off "
             "(the encoder must read raw ray-traced pixels, and 64x64 is below DLSS's minimum input "
             "anyway).\nSource: runs/spike/fps_benchmark.log, spike/out/{rgb,seg}_e64_r64.npy, "
             "spike/spike_fps_benchmark.py.",
             ha="center", va="bottom", fontsize=8.7, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.885, bottom=0.175)

    out = Path(args.out_dir) / "diagnostics" / "fig2b_throughput.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    for r in rows:
        print(f"  num_envs={r['num_envs']:>2}  env_steps/s={r['env_sps']:>7.1f}  "
              f"sim_steps/s={r['sim_sps']:>5.1f}  {r['gpu_day']:.1f}M/GPU-day")


if __name__ == "__main__":
    main()
