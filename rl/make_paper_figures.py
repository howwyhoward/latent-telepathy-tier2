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
import runpy
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

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
    "handoff_analysis.png": "sim2real_transfer",
}

# Per-figure text upscale applied to the PAPER copy only. These figures were
# designed 11-21 in wide; printed at IEEE full width (7.16 in) their text
# lands at 3.5-6 pt. The boost brings annotations to >=6.5 pt effective while
# staying small enough not to wreck hand-tuned layouts (verified visually).
FONT_BOOST = {
    "exploration_collapse": 1.45,
    "composition_check": 1.40,
    "exploration_sweep": 1.15,  # denser annotations; 1.35 caused collisions
    "occlusion_gate": 1.30,
    "conditions_ladder": 1.30,
    "pipeline": 1.30,
    "throughput": 1.30,
    "recruited_misused": 1.35,
    "positive_control": 1.20,
    "substrate": 1.20,
    "sim2real_transfer": 1.20,
}

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
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral",
                       "DejaVu Serif"],
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
    boost = FONT_BOOST.get(clean, 1.0)
    if boost > 1.0:
        from matplotlib.text import Text
        for t in self.findobj(Text):
            t.set_fontsize(t.get_fontsize() * boost)
    OUT.mkdir(parents=True, exist_ok=True)
    _orig_savefig(self, OUT / f"{clean}.pdf", bbox_inches="tight")
    _orig_savefig(self, OUT / f"{clean}.png", dpi=300, bbox_inches="tight")
    print(f"  paper: {clean}.pdf + .png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="run just the script whose filename contains this")
    args = p.parse_args()

    apply_style()
    Figure.savefig = _paper_savefig

    failures = []
    for rel in SCRIPTS:
        if args.only and args.only not in rel:
            continue
        print(f"== {rel}")
        sys.argv = [rel]
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


if __name__ == "__main__":
    main()
