"""Diagnostic figures 4, 7, 8 — the positive control and the six-null chain.

Same visual language as rl/plot_v8.py and Tier 1's rl/plot_m10.py, so all
three tiers of figure read as one series: thesis red #C44E52, controls in
blue/gray, gates as dashed reference lines, per-seed or per-run dots over
translucent means.

Isaac-free: every number is read from a committed CSV or diagnostic log. Figure 7
also draws the map, so it imports chokepoint.geometry, which needs the Tier 1
checkout beside this one (see TIER1 below) but never launches Kit.

  Figure 4  runs/archive/m7_navsolo/m7*.csv       the pixel positive control
  Figure 7  runs/diag/exploration{,_win}.log      route != action exploration
  Figure 7b the same logs                         the sigma-sweep parametrics
  Figure 8  runs/race_v7/*.csv                    coverage, then optimized away
            runs/diag/route_choice_v7oracle.log   the lie test

Usage:
    python rl/plot_diagnostics.py --out-dir plots
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 130
RED = "#C44E52"       # thesis condition
BLUE = "#4C72B0"      # ceiling / oracle
GRAY = "#bbbbbb"      # floor / silence
DARK = "#333333"
ORANGE = "#DD8452"
PURPLE = "#8172B2"
GREEN = "#55A868"

# v7 launch config, from runs/race_v7/oracle.json
V7 = dict(explore_log_std=1.5, explore_window=40, explore_tau=30.0,
          explore_anneal_frac=0.6, n_iters=366, base_log_std=-0.5088)
# geometry of the decision, from spike/diag_exploration.py's docstring
LATERAL_NEEDED = 1.3     # sustained body-frame vy to reach the far mouth
WINDOW = 30              # steps of the corridor-choice window

REPO = Path(__file__).resolve().parents[1]
TIER1 = REPO.parent / "latent-telepathy"   # chokepoint.geometry reads envs.constants


# -- shared helpers -----------------------------------------------------------

def read_csv(path):
    """CSV -> dict of column name -> float array, with '' and nan preserved."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def smooth(y, k=5):
    """Centred running mean that tolerates nan (the first logging row is nan)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        w = y[lo:hi]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def parse_sweep(path, columns):
    """Pull the fixed-width sweep table out of an Isaac log.

    The tables are printed after Kit's startup noise and can be interleaved
    with carb warnings, so rows are matched by shape: `columns` numeric fields
    (with 'dims' as the one string field) and nothing else on the line.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        toks = line.split()
        if len(toks) != len(columns):
            continue
        rec = {}
        for name, tok in zip(columns, toks):
            if name == "dims":
                if tok not in ("all", "y"):
                    rec = None
                    break
                rec[name] = tok
            else:
                try:
                    rec[name] = float(tok)
                except ValueError:
                    rec = None
                    break
        if rec:
            rows.append(rec)
    if not rows:
        raise SystemExit(f"no sweep rows parsed from {path}")
    return rows


# -- Figure 4: the pixel positive control -------------------------------------

def plot_m7(run_dir: Path, out: Path):
    runs = [
        ("m7", "m7 — Euclidean shaping", GRAY, "-"),
        ("m7b", "m7b — undertrained (2M)", ORANGE, "-"),
        ("m7c", "m7c — hazard priced at -0.5", PURPLE, "-"),
        ("m7e", "m7e — hazard at -0.05  PASS", RED, "-"),
    ]
    data = {tag: read_csv(run_dir / f"{tag}.csv") for tag, *_ in runs}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    (ax_s, ax_h), (ax_l, ax_b) = axes

    for tag, label, color, ls in runs:
        d = data[tag]
        x = d["global_step"] / 1e6
        lw = 2.2 if tag == "m7e" else 1.5
        z = 3 if tag == "m7e" else 2
        ax_s.plot(x, smooth(d["success"]), ls, color=color, lw=lw, label=label, zorder=z)
        ax_h.plot(x, smooth(d["hazard_steps"]), ls, color=color, lw=lw, zorder=z)
        ax_l.plot(x, smooth(d["ep_len"]), ls, color=color, lw=lw, zorder=z)

    ax_s.axhline(0.80, color=DARK, ls="--", lw=1.2, zorder=1)
    # right end of the line: the upper-left legend covers the left end entirely
    ax_s.text(0.985, 0.815, "gate 0.80", transform=ax_s.get_yaxis_transform(),
              fontsize=9, color=DARK, va="bottom", ha="right")
    ax_s.set_ylabel("success rate", fontsize=11)
    ax_s.set_ylim(-0.03, 1.03)
    ax_s.set_title("(a) can PPO solve this from pixels at all?", fontsize=11, loc="left")
    ax_s.legend(fontsize=8.5, loc="upper left", framealpha=0.92)

    m7e_haz = np.nanmean(data["m7e"]["hazard_steps"][-20:])
    ax_h.axhspan(21, 25, color=RED, alpha=0.10, zorder=0)
    ax_h.text(0.03, 30.5, "no-comms floor, 21-25 steps/episode",
              transform=ax_h.get_yaxis_transform(), ha="left", va="bottom",
              fontsize=9.5, color=RED)
    ax_h.set_ylabel("hazard steps / episode", fontsize=11)
    ax_h.set_ylim(-1.5, 44)
    ax_h.set_title("(b) the price a blind navigator pays", fontsize=11, loc="left")
    ax_h.annotate(f"m7e plateaus at {m7e_haz:.1f} —\nz_t later drives this to 0.00",
                  xy=(2.55, m7e_haz), xytext=(1.30, 8.0), fontsize=9, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))

    ax_l.set_ylabel("episode length (steps)", fontsize=11)
    ax_l.set_xlabel("environment steps (millions)", fontsize=11)
    ax_l.set_title("(c) refusal is visible as a longer episode", fontsize=11, loc="left")
    ax_l.annotate("m7 ran a 300-step cap; the geodesic fix\n"
                  "extended episodes to 600 (60 s)",
                  xy=(1.72, 300), xytext=(1.30, 180), fontsize=8.8, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
    ax_l.set_ylim(120, 660)

    # slab-side split: the refusal diagnosis, and what fixed it. Final logged
    # row, the same convention the report's numbers use.
    split = [("m7c", "m7c\nhazard -0.5", PURPLE), ("m7e", "m7e\nhazard -0.05", RED)]
    w, xs = 0.34, np.arange(len(split))
    for j, (side, hatch) in enumerate([("success_slab_top", None),
                                       ("success_slab_bottom", "///")]):
        vals = [data[t][side][-1] for t, *_ in split]
        ax_b.bar(xs + (j - 0.5) * w, vals, w, hatch=hatch,
                 color=[c for _, _, c in split], edgecolor="white",
                 alpha=0.55 if j else 0.9,
                 label="slab top" if j == 0 else "slab bottom")
        for xi, v in zip(xs + (j - 0.5) * w, vals):
            ax_b.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=9.5, color=DARK)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([lab for _, lab, _ in split], fontsize=10)
    ax_b.set_ylabel("success by slab side (final log)", fontsize=11)
    ax_b.set_ylim(0, 1.30)
    ax_b.set_xlim(-0.55, 1.55)
    ax_b.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)
    ax_b.set_title("(d) one-sided refusal, then balance", fontsize=11, loc="left")

    for ax in (ax_s, ax_h, ax_l):
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlim(0, 3.05)
    ax_b.grid(alpha=0.25, lw=0.6, axis="y")
    ax_h.set_xlabel("environment steps (millions)", fontsize=11)

    fig.suptitle("Figure 4 — M7 pixel positive control: three failures, each one a design constraint",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.012,
             "Single-agent PPO on 64x64 RGB, no communication. In (d), m7c's hazard price of -0.5/step made "
             "crossing cost twice the success bonus, so refusing the slabbed\ncorridor was rational; -0.05 "
             "restores Tier 1's ratio. m7d (a repeat of m7c after a warm-start attempt) is omitted: a refusing "
             "policy never samples the slab, so it\nnever observes the new price. "
             "Source: runs/archive/m7_navsolo/m7*.csv.",
             ha="center", va="bottom", fontsize=8.6, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    for tag in ("m7c", "m7e"):
        d = data[tag]
        print(f"  {tag} final: success {d['success'][-1]:.2f} "
              f"(top {d['success_slab_top'][-1]:.2f} / "
              f"bottom {d['success_slab_bottom'][-1]:.2f}), "
              f"hazard {d['hazard_steps'][-1]:.2f}")


# -- Figure 7: route exploration is not action exploration --------------------
#
# Rebuilt 2026-08-10 for a first-time reader. The earlier version was three
# abstract plots using sigma, tau, AR(1) and "vy" without ever showing the
# situation those symbols describe. The order now is: what the robot has to do,
# why chance never does it, what fixed it.

STEP_DT = 0.1          # env step, seconds (rendering step-size in the env)
LATERAL_LIMIT = 1.0    # cmd_vel is normalized to [-1, 1]


SLATE = "#6b7280"      # the uninteresting baseline; darker than GRAY so it prints
WALL = "#9aa2ac"
FREE = "#fafbfc"


def _map_image(grid):
    """Grid -> RGB image. EMPTY=1, WALL=2, HAZARD=5 (Tier 1's encoding).

    Hazard cells are painted as free space and the slab is overlaid as a patch, so
    that the slab as built and the ghost on the other side are drawn identically.
    """
    img = np.zeros((*grid.shape, 3))
    img[grid == 1] = matplotlib.colors.to_rgb(FREE)
    img[grid == 2] = matplotlib.colors.to_rgb(WALL)
    img[grid == 5] = matplotlib.colors.to_rgb(FREE)
    return img


def _box(ec="#c3c9d2", pad=0.34):
    return dict(boxstyle=f"round,pad={pad}", fc="white", ec=ec, lw=1.0, alpha=0.97)


def _draw_situation(ax):
    """Panel 1: the decision, drawn from the real scene geometry."""
    for p in (REPO, TIER1):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from chokepoint.geometry import chokepoint_grid, compute_geometry

    grid = chokepoint_grid(2)
    geo = compute_geometry(grid, 0.5)
    half = geo.size * geo.cell / 2
    ax.imshow(_map_image(grid), extent=(-half, half, -half, half), origin="upper",
              interpolation="nearest", zorder=0)

    nx, ny, _ = geo.starts["navigator"]
    x0, x1, ty0, ty1 = geo.corridor_top
    _, _, by0, by1 = geo.corridor_bottom
    top_y = (ty0 + ty1) / 2

    for (y0, y1), color in (((ty0, ty1), RED), ((by0, by1), BLUE)):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color, alpha=0.20,
                                   ec="none", zorder=1))

    # the slab as built this episode, and where it lands on the other draw
    hx0, hx1, hy0, hy1 = geo.hazard_aabb_top
    ax.add_patch(plt.Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fc="#4b5563",
                               ec="#374151", lw=1.2, zorder=3))
    ax.text((hx0 + hx1) / 2, (hy0 + hy1) / 2, "hazard\nslab", color="white",
            fontsize=9.6, ha="center", va="center", fontweight="bold", zorder=4,
            linespacing=1.3)
    gx0, gx1, gy0, gy1 = geo.hazard_aabb_bottom
    ax.add_patch(plt.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fc="none",
                               ec="#4b5563", lw=1.4, ls=(0, (3, 2.5)), zorder=3))
    ax.text((gx0 + gx1) / 2, (gy0 + gy1) / 2, "or here", color="#4b5563",
            fontsize=9.4, ha="center", va="center", fontweight="bold", zorder=4,
            style="italic")

    # the tint carries the route; the label sits alone inside its own band
    for y0, y1, color, label in ((ty0, ty1, RED, "NEVER taken\n0 of 128 episodes"),
                                 (by0, by1, BLUE,
                                  "ALWAYS taken\n128 of 128 episodes")):
        # -1.35 is the window between the sideways-travel arrow on the left and
        # the dashed alternative-slab box on the right, both of which this label
        # collided with at the narrower figure width
        ax.text(-1.35, (y0 + y1) / 2, label, color=color, fontsize=10.4,
                ha="center", va="center", fontweight="bold", zorder=6,
                linespacing=1.45)

    # what switching would cost, annotated up in the empty wall band
    ax.annotate("", xy=(nx + 0.42, top_y - 0.10), xytext=(nx + 0.42, ny),
                arrowprops=dict(arrowstyle="<|-|>", color=DARK, lw=1.9,
                                mutation_scale=12), zorder=6)
    ax.annotate("switching corridors costs\n"
                f"{top_y - ny:.2f} m of sideways travel —\n"
                "about 3 seconds of pushing",
                xy=(nx + 0.50, (ny + top_y) / 2 + 0.30), xytext=(-2.05, ty1 + 1.62),
                fontsize=10.2, color=DARK, ha="center", va="center", zorder=7,
                linespacing=1.55, bbox=_box(),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.4,
                                connectionstyle="arc3,rad=0.12"))

    ax.scatter([nx], [ny], s=210, color=DARK, marker="o", ec="white", lw=1.8,
               zorder=7)
    ax.text(nx - 0.34, ny, "start", fontsize=10.0, color=DARK, ha="right",
            va="center", zorder=7, fontweight="bold")

    ax.text(0.0, by0 - 0.72,
            # wrapped to three short lines: at two it was wider than the panel
            "The slab is redrawn every episode. When it\n"
            "lands in the bottom corridor, the only safe\n"
            "route is the one the robot never takes.",
            fontsize=10.2, color=DARK, ha="center", va="top", zorder=7,
            linespacing=1.55, bbox=_box())

    ax.set_xlim(-half + 0.35, half - 0.05)
    ax.set_ylim(by0 - 2.05, ty1 + 2.35)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _window_mean_std(sigma, n, rho):
    """Std of the mean of n AR(1) steps with marginal sigma (rho = 0 gives iid)."""
    k = np.arange(1, n)
    return float(np.sqrt(sigma ** 2 / n ** 2 * (n + 2 * np.sum((n - k) * rho ** k))))


def _draw_mechanism(ax, ax_dist, sigma, window=WINDOW, tau=30.0, n_paths=600, seed=0):
    """Panel 2: why chance never produces a sustained sideways push."""
    rng = np.random.default_rng(seed)
    rho = float(np.exp(-1.0 / tau))
    steps = np.arange(1, window + 1)

    def running_mean(correlated):
        z = rng.standard_normal((n_paths, window))
        if not correlated:
            a = z
        else:
            a = np.empty_like(z)
            a[:, 0] = z[:, 0]
            for t in range(1, window):
                a[:, t] = rho * a[:, t - 1] + (1 - rho ** 2) ** 0.5 * z[:, t]
        return np.cumsum(a * sigma, axis=1) / steps

    # widest band first, so the narrow one stays legible drawn over it
    for corr, color, n_show in ((True, BLUE, 3), (False, SLATE, 3)):
        paths = running_mean(corr)
        lo, hi = np.percentile(paths, [5, 95], axis=0)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.24, lw=0, zorder=2)
        ax.plot(steps, lo, color=color, lw=1.1, alpha=0.6, zorder=3)
        ax.plot(steps, hi, color=color, lw=1.1, alpha=0.6, zorder=3)
        ax.plot(steps, paths[:n_show].T, color=color, lw=1.0, alpha=0.85, zorder=4)

    s_iid = _window_mean_std(sigma, window, 0.0)
    s_ar1 = _window_mean_std(sigma, window, rho)

    for a in (ax, ax_dist):
        a.axhline(LATERAL_NEEDED, color=RED, lw=2.6, zorder=6)
        a.axhline(0, color="#c8cdd4", lw=1.0, zorder=1)
        a.grid(alpha=0.20, lw=0.6)
        a.set_ylim(-1.30, 2.02)
    ax.text(1.7, LATERAL_NEEDED + 0.13,
            f"+{LATERAL_NEEDED}  =  the sideways push\nneeded to switch corridors",
            ha="left", va="bottom", fontsize=10.0, color=RED, fontweight="bold",
            linespacing=1.5, zorder=7)

    ax.set_xlim(1, window)
    # wrapped: on one line this ran under the box-plot category labels to its right
    ax.set_xlabel(f"steps since entering the chamber\n({window} steps "
                  f"= {window * STEP_DT:.0f} seconds)", fontsize=10.8)
    ax.set_ylabel("sideways push, averaged over the steps so far", fontsize=10.8)

    for x, val, color, tx, ty in ((0.0, s_iid, SLATE, 0.02, 0.66),
                                  (1.0, s_ar1, BLUE, 0.88, 1.10)):
        ax_dist.add_patch(plt.Rectangle((x - 0.20, -val), 0.40, 2 * val, fc=color,
                                        alpha=0.55, ec=color, lw=1.4, zorder=4))
        ax_dist.plot([x, x], [-1.645 * val, 1.645 * val], color=color, lw=1.6,
                     zorder=3)
        for s in (-1, 1):
            ax_dist.plot([x - 0.09, x + 0.09], [s * 1.645 * val] * 2, color=color,
                         lw=1.6, zorder=3)
        gap = LATERAL_NEEDED / val
        ax_dist.text(tx, ty, f"{gap:.0f}x too small" if gap >= 10
                     else f"{gap:.1f}x too small", ha="center", va="center",
                     fontsize=9.8, color=color, fontweight="bold", zorder=7,
                     bbox=_box(ec=color, pad=0.3))
    ax_dist.set_xlim(-0.95, 1.95)
    ax_dist.set_xticks([0.0, 1.0])
    ax_dist.set_xticklabels(["drawn fresh\nevery step", "persists\nfor ~3 s"],
                            fontsize=10.0, fontweight="bold", linespacing=1.5)
    for tick, color in zip(ax_dist.get_xticklabels(), (SLATE, BLUE)):
        tick.set_color(color)
    ax_dist.tick_params(axis="y", labelleft=False, left=False)
    ax_dist.set_title("at the 3 s mark", fontsize=10.6, color=DARK, pad=10)
    return s_iid, s_ar1


def _draw_measurement(ax, iid_rows, win_rows):
    """Panel 3: the five noise settings that were actually measured."""
    def pick(rows, **kw):
        for r in rows:
            if all(abs(r[k] - v) < 1e-6 if isinstance(v, float) else r[k] == v
                   for k, v in kw.items()):
                return r
        raise SystemExit(f"no sweep row matching {kw}")

    trained = pick(iid_rows, log_std=-0.60, tau=0.0)
    louder = pick(iid_rows, log_std=0.0, tau=0.0)
    persists = pick(iid_rows, log_std=-0.60, tau=30.0)
    both = pick(iid_rows, log_std=0.50, tau=30.0)
    windowed = pick(win_rows, log_std=1.50, tau=30.0, win=40.0, dims="y")
    xn = trained["std"]
    configs = [
        (trained, "the policy's own noise   (all six races used this)",
         "not one episode even tried", RED),
        (louder, f"the same noise, turned up {louder['std'] / xn:.1f}x",
         "louder changes nothing", RED),
        (persists, "the same noise, made to persist for ~3 s",
         "persisting alone changes nothing", RED),
        (both, f"persistent and {both['std'] / xn:.1f}x louder, every axis, "
               "all episode",
         "finds the corridor,\nforgets how to drive", ORANGE),
        (windowed, f"persistent and {windowed['std'] / xn:.1f}x louder, "
                   "sideways only, first 4 s",
         "finds the corridor and\nkeeps the driving", GREEN),
    ]
    h = 0.30
    for i, (row, name, verdict, color) in enumerate(configs):
        y = -i
        ax.text(0.004, y + 0.50, name, fontsize=10.6, color=DARK, ha="left",
                va="center", fontweight="bold")
        for off, key, bcolor in ((h / 2 + 0.02, "top", RED),
                                 (-h / 2 - 0.02, "success", BLUE)):
            v = row[key]
            if v > 0:
                ax.barh(y + off, v, h, color=bcolor, ec="white", zorder=3)
            else:   # a zero bar still needs to read as a measurement, not a gap
                ax.plot([0, 0], [y + off - h / 2, y + off + h / 2], color=bcolor,
                        lw=2.6, solid_capstyle="butt", zorder=3)
            ax.text(v + 0.013, y + off, f"{v:.2f}", va="center", ha="left",
                    fontsize=10.2, color=bcolor, fontweight="bold", zorder=4)
        ax.text(1.08, y - 0.02, verdict, fontsize=10.2, color=color, ha="left",
                va="center", fontweight="bold", zorder=5, linespacing=1.5)
        if i:
            ax.plot([0, 2.12], [y + 0.74] * 2, color="#e6e9ed", lw=1.0, zorder=0)
    ax.plot([1.045, 1.045], [-4.55, 0.62], color="#d6dae0", lw=1.2, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=RED,
                             label="reached the other corridor"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE,
                             label="still finished the task")]
    ax.legend(handles=handles, fontsize=10.2, ncol=2, frameon=False,
              loc="lower left", bbox_to_anchor=(-0.004, -0.012),
              handlelength=1.3, columnspacing=2.0)
    ax.set_xlim(0, 2.12)
    ax.set_ylim(-5.05, 0.62)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks([])
    ax.set_xlabel("fraction of 128 test episodes", fontsize=10.8, x=0.26)
    ax.grid(alpha=0.20, lw=0.6, axis="x")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def plot_exploration_v2(diag_dir: Path, out: Path):
    iid = parse_sweep(diag_dir / "exploration.log",
                      ["log_std", "std", "tau", "top", "bottom", "neither", "success"])
    win = parse_sweep(diag_dir / "exploration_win.log",
                      ["log_std", "std", "tau", "win", "dims",
                       "top", "bottom", "neither", "success"])
    sigma = min(r["std"] for r in iid)   # the trained policy's own noise level

    fig = plt.figure(figsize=(13.2, 6.9))
    # the box-plot column carries two wide category labels for only two boxes,
    # so it needs more width than its data does; the bar chart has the slack
    gs = fig.add_gridspec(1, 4, width_ratios=[1.14, 0.72, 0.86, 1.12],
                          left=0.027, right=0.995, top=0.845, bottom=0.112,
                          wspace=0.26)
    ax_a, ax_b, ax_d, ax_c = (fig.add_subplot(gs[0, i]) for i in range(4))
    ax_d.set_position(ax_d.get_position().translated(-0.019, 0))

    _draw_situation(ax_a)
    s_iid, s_ar1 = _draw_mechanism(ax_b, ax_d, sigma)
    _draw_measurement(ax_c, iid, win)

    # step titles as figure text, so all three sit on one baseline whatever each
    # panel puts above its own axes
    for ax, step in ((ax_a, "1.  the decision"),
                     (ax_b, "2.  why it never happens by chance"),
                     (ax_c, "3.  what actually worked")):
        fig.text(ax.get_position().x0, 0.872, step, fontsize=13.5, ha="left",
                 va="bottom", fontweight="bold", color=DARK)

    fig.suptitle("Figure 7 — six experiments found nothing because the robot never "
                 "once tried the other corridor", fontsize=16.5, y=0.958)
    fig.text(0.027, 0.018,
             f"Panels 1 and 3 are measurements: 128 episodes per setting, "
             f"corridor-competent warm start, canonical start pose, seed-2 map. Panel 2 "
             f"is the arithmetic of the two noise processes at the policy's own noise "
             f"level (sigma = {sigma:.2f}, correlation time 3 s): box is +/-1 sd, "
             f"whiskers and bands the 5th-95th percentile over 600 draws, with three "
             f"sample paths each. Sources: runs/diag/exploration.log, "
             f"runs/diag/exploration_win.log.",
             ha="left", va="bottom", fontsize=8.8, color="#828a95")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  trained sigma {sigma:.2f}; {WINDOW}-step window-average std: "
          f"iid +/-{s_iid:.3f} ({LATERAL_NEEDED / s_iid:.1f}x short), "
          f"AR(1) +/-{s_ar1:.3f} ({LATERAL_NEEDED / s_ar1:.1f}x short)")


def plot_exploration(diag_dir: Path, out: Path):
    iid = parse_sweep(diag_dir / "exploration.log",
                      ["log_std", "std", "tau", "top", "bottom", "neither", "success"])
    win = parse_sweep(diag_dir / "exploration_win.log",
                      ["log_std", "std", "tau", "win", "dims",
                       "top", "bottom", "neither", "success"])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.4, 5.1))

    # (a) coverage only arrives once competence has left. tau=30 is the
    # correlation time v7 went on to use, so the correlated series is filtered
    # to it rather than averaged over the tau=100 rows.
    series = [
        ([r for r in iid if r["tau"] == 0], "iid (tau = 0)", DARK, "o"),
        ([r for r in iid if r["tau"] == 30], "correlated (tau = 30)", BLUE, "s"),
    ]
    for rows, label, color, marker in series:
        rows = sorted(rows, key=lambda r: r["std"])
        x = [r["std"] for r in rows]
        ax_a.plot(x, [r["top"] for r in rows], marker + "-", color=color, lw=2.2,
                  ms=7.5, label=f"alt-corridor rate, {label}")
        ax_a.plot(x, [r["success"] for r in rows], marker + "--", color=color,
                  lw=1.5, ms=6, alpha=0.45, mfc="white",
                  label=f"success rate, {label}")

    trained = min(iid, key=lambda r: r["std"])
    ax_a.scatter([trained["std"]], [trained["top"]], s=210, facecolor="none",
                 edgecolor=RED, lw=2.4, zorder=6)
    ax_a.annotate("the regime every race\nbefore v7 trained in:\n"
                  "0 of 128 episodes",
                  xy=(trained["std"], trained["top"]), xytext=(0.60, 0.26),
                  fontsize=9.5, color=RED, va="bottom",
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
                                  connectionstyle="arc3,rad=-0.25"))
    ax_a.set_xlabel("action-noise sigma", fontsize=11)
    ax_a.set_ylabel("rate over 128 episodes", fontsize=11)
    ax_a.set_ylim(-0.03, 0.95)
    ax_a.set_xlim(0.42, 1.78)
    ax_a.grid(alpha=0.25, lw=0.6)
    # raised: at 0.56 the frame clipped the tail of the red callout below it
    ax_a.legend(fontsize=8.2, loc="center right", bbox_to_anchor=(1.0, 0.66),
                framealpha=0.95, ncol=1)
    ax_a.set_title("(a) buying coverage with sigma destroys the policy first",
                   fontsize=11, loc="left")

    # (b) the arithmetic: how many sigma is the alternative route?
    sig = np.linspace(0.5, 4.7, 300)
    z_iid = LATERAL_NEEDED / (sig / np.sqrt(WINDOW))
    z_corr = LATERAL_NEEDED / sig
    ax_b.axhspan(5, 60, color=RED, alpha=0.07, zorder=0)
    ax_b.text(4.62, 20, "never sampled", ha="right", fontsize=10, color=RED)
    ax_b.plot(sig, z_iid, color=DARK, lw=2.6, zorder=3,
              label=f"iid noise: needs {LATERAL_NEEDED} lateral\n"
                    f"sustained across all {WINDOW} steps")
    ax_b.plot(sig, z_corr, color=BLUE, lw=2.6, zorder=3,
              label="AR(1), tau = 30: one deviation\nthat persists through the window")
    ax_b.axhline(2, color=GREEN, ls=":", lw=1.5, zorder=1)
    # mirrors "never sampled" above it; the legend is raised to clear it
    ax_b.text(4.62, 2.25, "merely rare", ha="right", fontsize=10, color=GREEN)

    z_at_trained = LATERAL_NEEDED / (trained["std"] / np.sqrt(WINDOW))
    z_corr_trained = LATERAL_NEEDED / trained["std"]
    ax_b.scatter([trained["std"]] * 2, [z_at_trained, z_corr_trained], s=80,
                 color=[DARK, BLUE], zorder=6)
    ax_b.annotate(f"at the trained sigma = {trained['std']:.2f}:\n"
                  f"a {z_at_trained:.0f}-sigma event",
                  xy=(trained["std"], z_at_trained), xytext=(1.55, 26),
                  fontsize=10, color=DARK,
                  arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax_b.annotate(f"the same per-step noise,\ncorrelated: {z_corr_trained:.1f} sigma",
                  xy=(trained["std"], z_corr_trained), xytext=(1.40, 0.265),
                  fontsize=10, color=BLUE,
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.22, 60)
    ax_b.set_xlim(0.5, 4.7)
    ax_b.set_yticks([0.3, 1, 3, 10, 30])
    ax_b.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax_b.set_xlabel("per-step action-noise sigma", fontsize=11)
    ax_b.set_ylabel("deviation required to reach the far mouth, in sigma",
                    fontsize=10.5)
    ax_b.grid(alpha=0.22, lw=0.6, which="major")
    # raised clear of the 2-sigma line and its label at the same right edge
    ax_b.legend(fontsize=8.2, loc="center right", bbox_to_anchor=(1.0, 0.62),
                framealpha=0.95)
    ax_b.set_title("(b) why: the fix is correlation, not scale", fontsize=11, loc="left")

    # (c) windowed + axis-restricted: coverage without paying for it.
    # Label offsets are explicit because two configurations nearly coincide.
    offsets = {  # (std, win, dims) -> (dx, dy, ha)
        (0.55, 0, "all"): (0, -16, "center"),
        (1.65, 0, "all"): (10, -3, "left"),
        (1.65, 40, "all"): (10, 4, "left"),
        (1.65, 40, "y"): (10, -14, "left"),
        (2.72, 40, "y"): (-11, -4, "right"),
        (2.72, 60, "y"): (10, -3, "left"),
    }
    best = max(win, key=lambda r: r["top"])
    for r in win:
        windowed = r["win"] > 0
        color = GREEN if windowed else DARK
        ax_c.scatter(r["success"], r["top"], s=155 if windowed else 115,
                     color=color, alpha=0.9 if windowed else 0.5,
                     marker="o" if r["dims"] == "y" else "s", zorder=4,
                     edgecolor="white", lw=1.0)
        key = (round(r["std"], 2), int(r["win"]), r["dims"])
        if key not in offsets:
            continue
        if r["win"] == 0 and r["std"] < 1.0:
            lab = f"{r['std']:.2f} · trained"
        elif r["win"] == 0:
            lab = f"{r['std']:.2f} · whole ep"
        else:
            lab = f"{r['std']:.2f} · w{int(r['win'])} · {r['dims'] if r['dims'] == 'y' else 'all'}"
        dx, dy, ha = offsets[key]
        ax_c.annotate(lab, xy=(r["success"], r["top"]), xytext=(dx, dy),
                      textcoords="offset points", fontsize=8.4, color=DARK, ha=ha)

    ax_c.scatter([best["success"]], [best["top"]], s=310, facecolor="none",
                 edgecolor=GREEN, lw=2.3, zorder=5)
    ax_c.annotate(f"the v7 configuration\nsigma {best['std']:.2f}, first "
                  f"{int(best['win'])} steps, vy only:\n{best['top']:.2f} coverage "
                  f"AND {best['success']:.2f} success",
                  xy=(best["success"], best["top"]), xytext=(0.045, 0.185),
                  fontsize=9.5, color=GREEN,
                  arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=DARK, alpha=0.5, ms=9,
                   label="whole-episode boost"),
        plt.Line2D([], [], marker="o", ls="", color=GREEN, ms=9,
                   label="windowed boost"),
    ]
    # lower left is the only free quadrant: the wide v7 callout fills the top
    # and the scatter labels run along the lower right
    ax_c.legend(handles=handles, fontsize=8.4, loc="lower left", framealpha=0.95)
    ax_c.set_xlabel("success rate          "
                    "(labels: sigma · boost window · axes)", fontsize=10.5)
    ax_c.set_ylabel("alternative-corridor sampling rate", fontsize=11)
    ax_c.set_xlim(0.02, 0.78)
    ax_c.set_ylim(-0.048, 0.27)
    ax_c.grid(alpha=0.25, lw=0.6)
    ax_c.set_title("(c) windowed, lateral-only AR(1) gets both", fontsize=11, loc="left")

    fig.suptitle("Figure 7 — route exploration is not action exploration: the cause of six nulls",
                 fontsize=13.5, y=0.975)
    fig.text(0.5, 0.035,
             "128 episodes per configuration from the canonical start, warm-started corridor-competent "
             "navigator. Tier 1 never met this: one gridworld action moved a whole\ncell, so route exploration "
             "and action exploration were the same operation. The sigma multiplier in (b) is order-of-magnitude "
             "and moves with which axis's sigma is taken.\nSource: runs/diag/exploration.log, "
             "runs/diag/exploration_win.log, spike/diag_exploration.py.",
             ha="center", va="bottom", fontsize=8.6, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.115, 1, 0.945])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  trained sigma {trained['std']:.2f}: alt-corridor {trained['top']:.2f}, "
          f"iid requirement {z_at_trained:.1f} sigma, AR(1) requirement "
          f"{LATERAL_NEEDED / trained['std']:.1f} sigma")
    print(f"  best windowed: sigma {best['std']:.2f} win {int(best['win'])} "
          f"dims {best['dims']} -> top {best['top']:.2f} at success {best['success']:.2f}")


# -- Figure 8: recruited, but for the wrong function --------------------------

def boost_sigma(iters):
    """v7's exploration-boost schedule, from rl/train_race.py.

    log-linear from explore_log_std down to the policy's own log_std over
    explore_anneal_frac of training. The base is held at its warm-start value
    for this overlay (the trainer uses the live value, which is not logged).
    """
    k = np.clip((iters - 1) / (V7["explore_anneal_frac"] * V7["n_iters"]), 0, 1)
    return np.exp((1 - k) * V7["explore_log_std"] + k * V7["base_log_std"])


def parse_lie_test(path):
    """Pull the two 2x2 route-choice tables out of the diagnostic log."""
    blocks = {}
    key = None
    for line in Path(path).read_text().splitlines():
        if "TRUE BIT" in line:
            key = "true"
            blocks[key] = {}
        elif "FLIPPED" in line:
            key = "lied"
            blocks[key] = {}
        elif key:
            m = re.match(r"\s+(top|bottom)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                         line)
            if m:
                blocks[key][m.group(1)] = dict(
                    to_top=float(m.group(2)), to_bottom=float(m.group(3)),
                    neither=float(m.group(4)), success=float(m.group(5)))
    if set(blocks) != {"true", "lied"} or len(blocks["true"]) != 2:
        raise SystemExit(f"could not parse the lie test from {path}")
    return blocks


def plot_v7(race_dir: Path, diag_dir: Path, out: Path):
    conds = [("oracle", "oracle (noiseless bit)", BLUE),
             ("z_t", "z_t (latent)", RED),
             ("none", "none (silence)", GRAY)]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.9, 4.95),
                                     gridspec_kw={"width_ratios": [1.25, 1]})

    early = {}
    for tag, label, color in conds:
        d = read_csv(race_dir / f"{tag}.csv")
        it = np.arange(1, len(d["success_slab_bottom"]) + 1)
        y = smooth(d["success_slab_bottom"], k=5)
        ax_l.plot(it, y, color=color, lw=2.1, label=label)
        m = (it >= 5) & (it <= 40)
        early[tag] = (np.nanmax(y[m]), np.nanmean(y[m]))

    ax_b = ax_l.twinx()
    it = np.arange(1, V7["n_iters"] + 1)
    ax_b.plot(it, boost_sigma(it), color=GREEN, lw=1.8, ls="--")
    ax_b.axhline(np.exp(V7["base_log_std"]), color=GREEN, lw=1.0, ls=":")
    ax_b.set_ylabel("exploration-boost sigma (lateral, first 40 steps)",
                    fontsize=10.5, color=GREEN)
    ax_b.tick_params(axis="y", colors=GREEN, labelsize=9)
    ax_b.set_ylim(0, 5.0)
    ax_b.text(196, np.exp(V7["base_log_std"]) + 0.13,
              f"learned sigma = {np.exp(V7['base_log_std']):.2f}",
              fontsize=8.5, color=GREEN)

    i80 = boost_sigma(np.array([80]))[0]
    ax_l.axvline(80, color=DARK, ls=":", lw=1.2)
    ax_l.annotate(
        f"by iteration 80 the alternative\n"
        f"route is gone, while the boost is\n"
        f"still sigma = {i80:.1f} — "
        f"{i80 / np.exp(V7['base_log_std']):.1f}x the learned noise",
        xy=(84, 0.012), xytext=(118, 0.145), fontsize=9.5, color=DARK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    ax_l.set_xlabel("PPO iteration", fontsize=11)
    ax_l.set_ylabel("success on the slab-bottom side\n(the side needing the far corridor)",
                    fontsize=11)
    ax_l.set_xlim(0, V7["n_iters"])
    ax_l.set_ylim(0, 0.30)
    ax_l.grid(alpha=0.25, lw=0.6)
    ax_l.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax_l.set_title("(a) coverage achieved, then optimized away", fontsize=11, loc="left")

    # the lie test
    lie = parse_lie_test(diag_dir / "route_choice_v7oracle.log")
    cols = [("true", "bit tells the truth"), ("lied", "bit is flipped")]
    rows = [("top", "slab in TOP\n(bottom is correct)"),
            ("bottom", "slab in BOTTOM\n(top is correct)")]
    for j, (ck, clab) in enumerate(cols):
        for i, (rk, rlab) in enumerate(rows):
            c = lie[ck][rk]
            chose_bottom = c["to_bottom"]
            fill = plt.get_cmap("Blues")(0.18 + 0.72 * c["success"])
            ax_r.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=fill,
                                         edgecolor="white", lw=3, alpha=0.55))
            ax_r.text(j + 0.5, 1 - i + 0.66,
                      f"route: {chose_bottom:.0%} bottom", ha="center",
                      fontsize=10.5, color=DARK, fontweight="bold")
            ax_r.text(j + 0.5, 1 - i + 0.40, f"success {c['success']:.2f}",
                      ha="center", fontsize=11, color=DARK)
            ax_r.text(j + 0.5, 1 - i + 0.17,
                      "correct route" if (rk == "top") else "wrong route",
                      ha="center", fontsize=9, color="#666666", style="italic")
    ax_r.set_xlim(0, 2)
    ax_r.set_ylim(0, 2)
    ax_r.set_xticks([0.5, 1.5])
    ax_r.set_xticklabels([c[1] for c in cols], fontsize=10.5)
    ax_r.set_yticks([1.5, 0.5])
    ax_r.set_yticklabels([r[1] for r in rows], fontsize=10)
    ax_r.tick_params(length=0)
    for s in ax_r.spines.values():
        s.set_visible(False)
    ax_r.set_title("(b) the lie test: the bit moved success, never the route",
                   fontsize=11, loc="left")
    ax_r.text(1.0, -0.24,
              "Deterministic rollouts of the v7 oracle policy. The corridor is the same in all "
              "four cells,\nso the message was gating advance-versus-balk INSIDE the corridor it "
              "had already chosen.",
              ha="center", va="top", fontsize=8.8, color="#555555", linespacing=1.5)

    fig.suptitle("Figure 8 — generation 7: the channel was recruited, but for the wrong function",
                 fontsize=13.5, y=0.972)
    fig.tight_layout(rect=[0, 0.045, 1, 0.935])
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    for tag, (mx, mean) in early.items():
        print(f"  {tag}: slab-bottom over iters 5-40  max {mx:.2f}, mean {mean:.2f}")
    print(f"  boost sigma at iter 80: {i80:.2f} "
          f"({i80 / np.exp(V7['base_log_std']):.2f}x learned)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="plots")
    ap.add_argument("--runs-dir", type=str, default="runs")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    out = Path(args.out_dir) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    plot_m7(runs / "archive" / "m7_navsolo", out / "fig4_m7_positive_control.png")
    plot_exploration_v2(runs / "diag", out / "fig7_exploration_collapse.png")
    # the sigma-sweep version: same data, for a reader who wants the parametrics
    plot_exploration(runs / "diag", out / "fig7b_exploration_sweep.png")
    plot_v7(runs / "race_v7", runs / "diag", out / "fig8_v7_recruited_misused.png")


if __name__ == "__main__":
    main()
