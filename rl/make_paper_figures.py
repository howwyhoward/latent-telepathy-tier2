"""Regenerate every report figure as a publication asset in plots/paper/.

Runs the existing figure scripts unmodified and intercepts their saves:
the legacy report PNG is written exactly as before, and a second copy is
written to plots/paper/ as <clean_name>.pdf (vector, fonts embedded as
TrueType per IEEE requirements) plus <clean_name>.png at 300 dpi, with the
narrative suptitle stripped -- in a paper that sentence belongs in the
caption, not baked into the image.

    python rl/make_paper_figures.py            # all figures
    python rl/make_paper_figures.py --only plot_v8.py
"""

import argparse
import re
import runpy
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "paper"

# legacy basename -> clean paper name (no figure numbering: numbers live in
# the LaTeX source, filenames should survive reordering)
RENAMES = {
    "fig1_substrate.png": "substrate",
    "fig_tier2_pipeline.png": "pipeline",
    "fig_tier2_conditions.png": "conditions_ladder",
    "fig2b_throughput.png": "throughput",
    "fig3_occlusion_gate.png": "occlusion_gate",
    "fig4_m7_positive_control.png": "positive_control",
    "fig7_exploration_collapse.png": "exploration_collapse",
    "fig7b_exploration_sweep.png": "exploration_sweep",
    "fig8_v7_recruited_misused.png": "recruited_misused",
    "fig8b_composition_check.png": "composition_check",
    "jepa_training.png": "encoder_training",
    "jepa_probes.png": "encoder_probes",
    "obey_gate_curves.png": "obedience_gate",
    "v8_race_curves.png": "race_curves",
    "v8_race_seed_bars.png": "route_optimality",
    "v8_success_seed_bars.png": "task_success",
    "v8_hazard_bars.png": "hazard_exposure",
    "v8_entropy_curves.png": "decision_entropy",
    "v8_corruption_bars.png": "corruption_controls",
    "v8b_sweep_bars.png": "condition_sweep",
    "sim_vs_real_frames.png": "sim2real_transfer",
}

IEEE_FULL_WIDTH_IN = 7.16   # two-column text width, i.e. figure* placement
TARGET_EFFECTIVE_PT = 6.5   # legibility floor for annotations once printed
MAX_BOOST = 1.25            # past this, the hand-tuned layouts start colliding
TITLE_PAD_PT = 7.0          # keeps panel content off its own axes title


def _title_pad_pt(ax):
    """Current axes-title pad in points, read back off the title's transform."""
    try:
        dy = (ax.title.get_transform().transform((0.0, 1.0))[1]
              - ax.transAxes.transform((0.0, 1.0))[1])
    except Exception:
        return 0.0
    return dy / ax.figure.dpi * 72.0


def _set_title_pad_pt(ax, pad_pt):
    """Offset an axes title by `pad_pt`.

    Axes.set_title(pad=...) would reset the title's font properties to the
    rcParam defaults and re-centre a loc="left" title, discarding the
    per-panel sizes these scripts set deliberately. Composing the same
    translation matplotlib uses internally moves the title and nothing else.
    """
    off = ScaledTranslation(0.0, pad_pt / 72.0, ax.figure.dpi_scale_trans)
    ax.title.set_transform(ax.transAxes + off)
    ax.title.set_clip_box(None)


def _legibility_pass(fig):
    """Raise only the SMALL text toward the print legibility floor.

    These figures were designed 6-21 in wide, so at IEEE full width their text
    shrinks by design_width / 7.16. Scaling every string by one per-figure
    factor (the first attempt) grew titles and long annotations into their own
    panel content -- visible as value labels touching category labels in
    exploration_collapse and bar labels touching panel titles in
    composition_check. Clamping each string up to the floor instead, capped at
    MAX_BOOST, fixes the unreadably small text and leaves already-large text
    exactly where it was designed to sit.

    Returns the smallest surviving effective point size, so the caller can
    report the figures that still need a layout redesign rather than a rescale.
    """
    shrink = fig.get_figwidth() / IEEE_FULL_WIDTH_IN
    floor_pt = TARGET_EFFECTIVE_PT * shrink
    smallest = None
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        fs = t.get_fontsize()
        if fs < floor_pt:
            fs = min(floor_pt, fs * MAX_BOOST)
            t.set_fontsize(fs)
        eff = fs / shrink
        smallest = eff if smallest is None else min(smallest, eff)
    for ax in fig.axes:
        if ax.title.get_text().strip():
            _set_title_pad_pt(ax, max(_title_pad_pt(ax), TITLE_PAD_PT))
    return smallest

# the deployment figure lives outside rl/ and needs the realcam20 checkpoints
# named explicitly (its defaults are the original-camera ones)
HANDOFF = ("handoff/analyze_handoff.py", [
    "--jepa_ckpt", "checkpoints/jepa_realcam20.pt",
    "--probe", "checkpoints/slab_probe_realcam20.pt",
    "--policy", "export/policy_deploy_realcam20.pt",
    "--sim_data", "/data/howard/isaac/datasets/chokepoint_v3_realcam20.npz",
])

SCRIPTS = [
    "rl/plot_fig1.py",
    "rl/plot_fig_pipeline.py",
    "rl/plot_fig2_conditions.py",
    "rl/plot_fig2b_throughput.py",
    "rl/plot_fig3_occlusion.py",
    "rl/plot_diagnostics.py",
    "rl/plot_fig8b_composition.py",
    "rl/plot_jepa.py",
    "rl/plot_v8.py",
]


def apply_style() -> None:
    """Publication rcParams: serif text matching an IEEE body, embeddable fonts.

    fonttype 42 embeds TrueType outlines instead of Type-3 bitmaps, which is
    what IEEE PDF eXpress checks for. Explicit per-call fontsizes inside the
    scripts are respected; only family/defaults change.

    Every fallback here must be a real TrueType (glyf) face. Nimbus Roman was
    in this list and is the default Times substitute on Linux, but it ships as
    CFF OpenType: fonttype 42 then emits a CFF stream behind a TrueType font
    descriptor, and validators report "mismatch between font type and embedded
    font file". Liberation Serif is TrueType and metrically Times-compatible.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Tinos",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


_orig_savefig = Figure.savefig


def _paper_savefig(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)  # legacy report output
    clean = RENAMES.get(Path(str(fname)).name)
    if clean is None:
        return
    if getattr(self, "_suptitle", None) is not None:
        self._suptitle.remove()
    # single-panel figures: the title belongs in the caption; multi-panel
    # figures keep their titles, which act as panel labels. A twinx pair is
    # still a single panel (two axes sharing one position).
    positions = {tuple(round(v, 4) for v in ax.get_position().bounds)
                 for ax in self.axes}
    if len(positions) == 1:
        for ax in self.axes:
            ax.set_title("")
    # bottom-margin figure texts are source/footnote paragraphs -- in a paper
    # that prose belongs in the caption, and after the font boost it collides
    # with panel annotations (seen on occlusion_gate)
    for t in list(self.texts):
        if t.get_position()[1] < 0.13:
            t.remove()
    # per-seed index labels (s1/s2/s3) sit between the dots and the mean value
    # and duplicate what the dots already show; dropping them is what buys the
    # mean labels their clearance
    for t in self.findobj(Text):
        if re.fullmatch(r"s\d+", t.get_text().strip()):
            t.set_visible(False)
    smallest = _legibility_pass(self)
    OUT.mkdir(parents=True, exist_ok=True)
    # pad_inches keeps a tight bbox from cropping flush against the outermost
    # label; 0.02 in is ~1.5 pt of margin at print size
    save_kw = dict(bbox_inches="tight", pad_inches=0.12)
    _orig_savefig(self, OUT / f"{clean}.pdf", **save_kw)
    _orig_savefig(self, OUT / f"{clean}.png", dpi=300, **save_kw)
    flag = "" if smallest is None or smallest >= 6.0 else "  << below 6 pt, needs redesign"
    print(f"  paper: {clean}.pdf + .png   min {smallest:.1f} pt at full width{flag}")


def check_embedded_fonts(paths):
    """Report any PDF whose fonts are not embedded TrueType.

    /FontFile2 is a TrueType stream; /FontFile3 is CFF (Type1C or OpenType)
    and /FontFile is Type 1. A missing FontFile means Type 3, which PDF
    eXpress rejects outright. Byte scanning is enough here and keeps this
    check dependency-free.
    """
    bad = {}
    for path in paths:
        blob = path.read_bytes()
        found = [m for m in (b"/FontFile3", b"/FontFile ", b"/FontFile\n",
                             b"/CIDFontType0") if m in blob]
        if b"/FontFile2" not in blob and b"/Font" in blob:
            found.append(b"no TrueType stream")
        if found:
            bad[path.name] = sorted({m.decode().strip() for m in found})
    if bad:
        print("\nNON-TRUETYPE FONT DATA (PDF eXpress will flag these):")
        for name, kinds in sorted(bad.items()):
            print(f"  {name}: {', '.join(kinds)}")
    else:
        print(f"fonts: all {len(paths)} PDFs embed TrueType only")
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="run just the script whose filename contains this")
    args = p.parse_args()

    apply_style()
    Figure.savefig = _paper_savefig

    failures = []
    for rel, extra in [(s, []) for s in SCRIPTS] + [HANDOFF]:
        if args.only and args.only not in rel:
            continue
        print(f"== {rel}")
        sys.argv = [rel] + extra
        try:
            runpy.run_path(str(ROOT / rel), run_name="__main__")
        except Exception:
            traceback.print_exc()
            failures.append(rel)
        finally:
            plt.close("all")

    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print(f"\nall paper figures in {OUT}")
    if check_embedded_fonts(sorted(OUT.glob("*.pdf"))):
        sys.exit(1)


if __name__ == "__main__":
    main()
