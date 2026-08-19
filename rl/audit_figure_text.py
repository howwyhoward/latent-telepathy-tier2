"""Geometric legibility audit for the paper figures.

Renders each figure and measures every visible Text artist's bounding box, so
crowding is reported as numbers rather than eyeballed. Three checks:

  OVERLAP      two text boxes intersect by more than `min_overlap_pt` in both
               directions
  CLEARANCE    a text box sits closer than `min_gap_pt` to an axes title
  BEHIND-BOX   a text box runs under an opaque patch drawn over it: a legend
               frame, or the bbox of another text. Those glyphs are hidden even
               though no two text boxes collide.

OVERLAP is measured in points rather than as a fraction of the smaller box.
Two labels that clip each other by a few glyph widths are a defect however
large the labels are, and the fractional test missed exactly that: a tick
label sitting on the x-axis label came to 15% and one abutting its neighbour
16%, both under an 18% threshold, while both were plainly broken in print.

Rotated text is skipped for OVERLAP (its axis-aligned bbox overstates extent,
which produces false positives against neighbouring row labels).

Three things must be excluded or every crowded-looking figure reports phantom
collisions, which is what the first version of this audit did:

  * Annotation.get_window_extent returns the union of the text and its arrow,
    so a callout whose arrow reaches into a neighbouring label was reported as
    a 64-100% overlap with it. Extents are measured as text only.
  * Tick labels whose tick lies outside the axis view interval still report a
    window extent but are never drawn, so a '-1.5' one step below the axis
    bottom appeared to sit on the x-axis label.
  * A text box outside the figure canvas is not cropped, because the export
    saves with bbox_inches="tight" -- the saved bbox grows to contain it. The
    old SPILL check therefore only ever fired on those phantom ticks.

    python rl/audit_figure_text.py            # audit every figure
    python rl/audit_figure_text.py --only exploration_collapse
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.text import Text        # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))


def _undrawn_tick_labels(fig):
    """ids of tick labels whose tick sits outside its axis view interval."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            try:
                lo, hi = sorted(axis.get_view_interval())
            except Exception:
                continue
            slack = (hi - lo) * 1e-6
            ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
            for tick in ticks:
                if lo - slack <= tick.get_loc() <= hi + slack:
                    continue
                hidden.update(id(lbl) for lbl in (tick.label1, tick.label2))
    return hidden


def _opaque(patch):
    alpha = patch.get_alpha()
    if alpha is not None and alpha < 0.5:
        return False
    return patch.get_visible() and (bool(patch.get_fill())
                                    or patch.get_edgecolor() is not None)


def _occluders(fig, items):
    """(patch, exempt_ids, zorder, what) for everything opaque drawn over text.

    A legend only hides text drawn below it, so its zorder is returned and
    compared. A text's own bbox is drawn with that text, and when it lands on a
    neighbouring label the two collide visually whichever ends up on top, so
    those are reported unconditionally (zorder None).
    """
    out = []
    legends = list(fig.legends)
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            legends.append(lg)
    for lg in legends:
        if lg.get_visible() and _opaque(lg.get_frame()):
            out.append((lg.get_frame(), {id(t) for t in lg.findobj(Text)},
                        lg.get_zorder(), "legend"))
    for t, _ in items:
        bp = t.get_bbox_patch()
        if bp is not None and _opaque(bp):
            out.append((bp, {id(t)}, None, f"box of '{t.get_text().split(chr(10))[0][:24]}'"))
    return out


def _boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = _undrawn_tick_labels(fig)
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in hidden:
            continue
        try:
            # Text.__func__ rather than t.get_window_extent: Annotation's
            # override unions in the arrow, which is not text and is allowed
            # to reach across the panel
            bb = Text.get_window_extent(t, renderer=r)
        except Exception:
            continue
        if bb.width <= 1 or bb.height <= 1:
            continue
        out.append((t, bb))
    return out


def audit(fig, name, min_overlap_pt=1.5, min_gap_pt=2.0, verbose=True):
    """Return a list of human-readable problems for one rendered figure."""
    items = _boxes(fig)
    dpi = fig.dpi
    gap_px = min_gap_pt * dpi / 72.0
    titles = {id(ax.title) for ax in fig.axes}
    measured = {id(t): b for t, b in items}
    problems = []

    def label(t):
        s = " ".join(t.get_text().split())
        return s[:46] + ("..." if len(s) > 46 else "")

    for i, (t1, b1) in enumerate(items):
        for t2, b2 in items[i + 1:]:
            if t1.get_rotation() % 180 or t2.get_rotation() % 180:
                continue
            ix = Bbox.intersection(b1, b2)
            if ix is None:
                continue
            w_pt, h_pt = ix.width / dpi * 72.0, ix.height / dpi * 72.0
            if w_pt >= min_overlap_pt and h_pt >= min_overlap_pt:
                kind = "TITLE-OVERLAP" if (id(t1) in titles or id(t2) in titles) else "OVERLAP"
                problems.append(f"{kind} {w_pt:.1f}x{h_pt:.1f} pt  "
                                f"'{label(t1)}'  vs  '{label(t2)}'")

    for ax in fig.axes:
        tb = ax.title
        bt = measured.get(id(tb))
        if bt is None or not tb.get_text().strip():
            continue
        for t, b in items:
            if t is tb or b.y1 <= bt.y0:
                continue
            horiz = min(b.x1, bt.x1) - max(b.x0, bt.x0)
            if horiz <= 0:
                continue
            gap = b.y0 - bt.y1
            if -gap_px < gap < gap_px:
                problems.append(
                    f"TIGHT-TO-TITLE {gap / dpi * 72:+.1f} pt  "
                    f"'{label(t)}'  under  '{label(tb)}'")

    r = fig.canvas.get_renderer()
    for patch, exempt, zorder, what in _occluders(fig, items):
        try:
            pb = patch.get_window_extent(renderer=r)
        except Exception:
            continue
        for t, b in items:
            if id(t) in exempt or (zorder is not None and t.get_zorder() > zorder):
                continue
            ix = Bbox.intersection(b, pb)
            if ix is None:
                continue
            w_pt, h_pt = ix.width / dpi * 72.0, ix.height / dpi * 72.0
            if w_pt >= min_overlap_pt and h_pt >= min_overlap_pt:
                problems.append(f"BEHIND-BOX {w_pt:.1f}x{h_pt:.1f} pt  "
                                f"'{label(t)}'  under {what}")

    if verbose:
        if problems:
            print(f"\n{name}: {len(problems)} issue(s)")
            for p in dict.fromkeys(problems):
                print(f"    {p}")
        else:
            print(f"{name}: clean")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    args = p.parse_args()

    import make_paper_figures as mpf

    findings = {}

    # audit writes nothing: replace the real save with a no-op so the committed
    # PNGs are untouched and the 300 dpi encodes (seconds each) are skipped
    mpf._orig_savefig = lambda *a, **kw: None

    def audited(self, fname, *a, **kw):
        clean = mpf.RENAMES.get(Path(str(fname)).name)
        if clean is None or (args.only is not None and args.only not in clean):
            return None
        mpf._paper_savefig(self, fname, *a, **kw)  # same transforms as the export
        findings[clean] = audit(self, clean)
        return None

    from matplotlib.figure import Figure
    Figure.savefig = audited
    mpf.apply_style()

    import runpy
    import matplotlib.pyplot as plt
    # HANDOFF as well as SCRIPTS: the deployment figure needs auditing like any
    # other, and it takes extra argv to find the realcam20 checkpoints
    for rel, extra in [(s, []) for s in mpf.SCRIPTS] + [mpf.HANDOFF]:
        sys.argv = [rel] + extra
        try:
            runpy.run_path(str(ROOT / rel), run_name="__main__")
        except Exception as exc:
            print(f"  {rel} failed: {exc}")
        finally:
            plt.close("all")

    bad = {k: v for k, v in findings.items() if v}
    print("\n" + "=" * 60)
    print(f"{len(findings) - len(bad)}/{len(findings)} figures clean")
    if bad:
        print("needs work: " + ", ".join(sorted(bad)))


if __name__ == "__main__":
    main()
