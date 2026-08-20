"""Phase 4 gate analysis of the 15 Aug robot handoff.

Consumes the exact (64,64,3) float32 arrays the robot fed the network and runs:
  1. zero-shot slab-probe transfer (the pre-registered WP7 measurement),
  2. latent health vs sim statistics,
  3. policy-output cross-check against the robot's policy_probe.csv,
  4. a side-by-side figure: sim scout frames vs real frames.

    /data/howard/isaac/envs/isaaclab/bin/python handoff/analyze_handoff.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chokepoint.jepa import PixelEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jepa_ckpt", default="checkpoints/jepa_pixels.pt")
ap.add_argument("--probe", default="checkpoints/slab_probe.pt")
ap.add_argument("--policy", default="export/policy_deploy.pt",
                help="TorchScript policy for the action cross-check; 'none' skips")
ap.add_argument("--sim_data", default="isaac-data/datasets/chokepoint_v1.npz")
ap.add_argument("--fig_out", default="handoff/sim_vs_real_frames.png")
ARGS = ap.parse_args()

HANDOFF = ROOT / "handoff/robomaster_handoff_20260815"

# label 1 = red pad (hazard analog) visible, 0 = not; None = health check only
FRAMES = {
    "ctrl_a": 0, "ctrl_b": 0,
    "red_100": 1, "red_150": 1, "red_200": 1, "red_280": 1,
    "red_l_280": 1, "red_r_280": 1, "rg_split_280": 1,
    "green_200": 0, "green_280": 0,
    "calib_tape_150": None, "calib_tape_280": None,
}


def main():
    device = "cpu"
    ck = torch.load(ROOT / ARGS.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).eval()
    encoder.load_state_dict(ck["encoder"])

    pk = torch.load(ROOT / ARGS.probe, map_location=device)
    w, b = pk["w"], pk["b"]
    sim_mu, sim_std = pk["sim_latent_mean"], pk["sim_latent_std"]

    policy = None if ARGS.policy == "none" else torch.jit.load(str(ROOT / ARGS.policy))
    r_top = torch.tensor([[1.0, 0.0]])

    names, imgs, labels = [], [], []
    for stem, lab in FRAMES.items():
        matches = sorted(HANDOFF.glob(f"pad_captures/{stem}_*_64.npy"))
        assert len(matches) == 1, f"{stem}: {matches}"
        a = np.load(matches[0])
        assert a.shape == (64, 64, 3) and a.dtype == np.float32
        names.append(stem)
        imgs.append(torch.from_numpy(a).permute(2, 0, 1))
        labels.append(lab)
    x = torch.stack(imgs)  # (13, 3, 64, 64)

    with torch.no_grad():
        z = encoder(x)
        logit = z @ w + b.squeeze()
        pred = (logit > 0).long()
        act = (policy(x, r_top.repeat(len(x), 1)) if policy is not None
               else torch.full((len(x), 3), float("nan")))

    print(f"{'frame':16s} {'label':>5s} {'probe':>5s} {'logit':>8s} "
          f"{'policy (route=top)':>26s}")
    for i, n in enumerate(names):
        lab = labels[i]
        ls = "-" if lab is None else str(lab)
        ok = ""
        if lab is not None:
            ok = "  OK" if int(pred[i].item()) == lab else "  MISS"
        a = act[i].tolist()
        print(f"{n:16s} {ls:>5s} {pred[i].item():>5d} {logit[i].item():>8.2f} "
              f"[{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]{ok}")

    keep = np.array([lab is not None for lab in labels])
    y = np.array([lab for lab in labels if lab is not None])
    lg = logit.numpy()[keep]
    pos, neg = lg[y == 1], lg[y == 0]
    pooled = float(((lg > 0) == y).mean())
    baseline = max(y.mean(), 1.0 - y.mean())
    tpr, tnr = float((pos > 0).mean()), float((neg <= 0).mean())
    balanced = (tpr + tnr) / 2
    auc = float((1.0 * (pos[:, None] > neg[None, :])
                 + 0.5 * (pos[:, None] == neg[None, :])).mean())
    grid = np.concatenate([lg - 1e-6, lg + 1e-6])
    best = max(float(((lg > t) == y).mean()) for t in grid)

    print(f"\nZERO-SHOT PROBE TRANSFER (pooled, {len(y)} labeled frames): "
          f"{pooled:.3f}")
    print("pre-registered rule: >=0.9 transfers | <=0.6 no transfer | else partial")
    # Pooled accuracy is only evidence above what a constant predictor scores.
    # This capture set is unbalanced, so quote the baseline and the
    # chance-corrected statistics next to the pre-registered number: a probe
    # that saturates on one class can otherwise clear the 0.6 boundary on class
    # balance alone.
    print(f"  constant-predictor baseline       {baseline:.3f} "
          f"({int(y.sum())} present / {int(len(y) - y.sum())} absent)")
    print(f"  per-class TPR {tpr:.3f} / TNR {tnr:.3f} -> balanced acc "
          f"{balanced:.3f}   (0.500 = chance)")
    print(f"  AUC, threshold-free ranking       {auc:.3f}   (0.500 = chance)")
    print(f"  best accuracy over all thresholds {best:.3f}"
          + ("  -> no threshold recovers the bit" if best <= baseline + 1e-9
             else "  -> recalibration would help"))
    verdict = ("TRANSFERS" if balanced >= 0.9
               else "NO TRANSFER" if balanced <= 0.6 else "PARTIAL")
    print(f"  same rule on chance-corrected accuracy: {verdict}")

    # latent health
    real_std = z.std(dim=0)
    ratio = real_std / (sim_std + 1e-8)
    maha = (((z - sim_mu) / (sim_std + 1e-8)) ** 2).mean(dim=1).sqrt()
    print(f"\nlatent health ({len(z)} real frames):")
    print(f"  per-dim std ratio real/sim: median {ratio.median():.2f} "
          f"min {ratio.min():.2f} max {ratio.max():.2f}")
    print(f"  dims collapsed (<5% sim std): {int((real_std < 0.05*sim_std).sum())}/64")
    print(f"  mean normalized distance from sim latent cloud: "
          f"{maha.mean():.2f} (in-distribution ~1.0)")

    # sim comparison: scout frames with/without slab from the JEPA dataset
    d = np.load(ROOT / ARGS.sim_data)
    rgb, seg, valid = d["rgb"], d["seg"], d["valid"]
    haz_cls = list(d["seg_classes"]).index("hazard")
    haz_px = (seg == haz_cls).sum(axis=(1, 2))
    cov = haz_px / (seg.shape[1] * seg.shape[2])

    def sim_logits(idx):
        sx = torch.from_numpy(rgb[idx]).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            sz = encoder(sx)
            return ((sz @ w + b.squeeze()).numpy(),
                    (((sz - sim_mu) / (sim_std + 1e-8)) ** 2).mean(dim=1)
                    .sqrt().numpy())

    # Reading the probe off a handful of frames with a large slab overstates it:
    # the response is graded in apparent size, so stratify. The real captures put
    # the pad at 2-21 % of the view, so the bands that matter are the small ones.
    print("\nsim probe response vs hazard coverage of the frame:")
    bands = [(0.0, 1e-9, "no hazard"), (1e-9, 0.02, "  0-2 %"),
             (0.02, 0.045, "  2-4.5 %"), (0.045, 0.09, "4.5-9 %"),
             (0.09, 0.25, "  9-25 %"), (0.25, 1.01, " 25-100 %")]
    band_mean = {}
    for lo, hi, tag in bands:
        sel = np.where(valid & (cov >= lo) & (cov < hi))[0][:200]
        if len(sel) == 0:
            continue
        lgs, _ = sim_logits(sel)
        band_mean[tag] = lgs.mean()
        print(f"  {tag:>10s}  n={len(sel):3d}  logit mean {lgs.mean():7.2f} "
              f"[{lgs.min():7.2f}, {lgs.max():7.2f}]")

    # The controls carry no pad at all, so their displacement from the sim
    # no-hazard band is pure domain nuisance. Compare it against what the pad
    # itself is worth at the coverage the real captures actually achieve.
    ctrl = float(logit.numpy()[[names.index("ctrl_a"), names.index("ctrl_b")]].mean())
    far = float(logit.numpy()[[names.index(n) for n in
                               ("red_280", "red_l_280", "red_r_280",
                                "rg_split_280")]].mean())
    none_mu, matched_mu = band_mean["no hazard"], band_mean["  2-4.5 %"]
    print("\nnuisance vs signal along the probe direction:")
    print(f"  sim, no hazard                      {none_mu:7.2f}")
    print(f"  sim, hazard at 2-4.5 % coverage     {matched_mu:7.2f}   "
          f"pad is worth {matched_mu - none_mu:+.2f} here")
    print(f"  real, empty arena (no pad)          {ctrl:7.2f}   "
          f"background alone shifts {ctrl - none_mu:+.2f}")
    print(f"  real, pad at 2-2.5 % coverage       {far:7.2f}   "
          f"pad is worth {far - ctrl:+.2f} (wrong sign)")
    print(f"  -> at the distances captured, the domain shift is "
          f"{abs(ctrl - none_mu) / max(abs(matched_mu - none_mu), 1e-6):.0f}x the "
          f"hazard signal the probe has to work with")

    # figure panels: match the sim row to the real row's apparent slab size, so
    # "close/mid/far" is a like-for-like comparison rather than three big slabs
    vis_idx = np.array([int(np.argmin(np.where(valid, abs(cov - t), 9e9)))
                        for t in (0.083, 0.041, 0.023)])
    non_idx = np.where(valid & (haz_px == 0))[0][:4]
    _, sim_maha = sim_logits(np.concatenate([vis_idx, non_idx]))
    print(f"\nsim frame distance from sim cloud: {sim_maha.mean():.2f}")

    # side-by-side figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAZ = "#c0392b"
    SAFE = "#5d6d7e"
    # sim exemplars matched to the real columns: hazard near, hazard far, none
    order = vis_idx[np.argsort(-cov[vis_idx])]
    sim_panels = [(i, f"slab {cov[i] * 100:.0f} % of view", HAZ) for i in order]
    sim_panels.append((non_idx[0], "no slab", SAFE))
    real_panels = [
        ("red_150", "red pad 1.5 m", HAZ),
        ("red_200", "red pad 2.0 m", HAZ),
        ("red_280", "red pad 2.8 m", HAZ),
        ("ctrl_a", "empty arena", SAFE),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(11.5, 6.4),
        # top leaves room for the suptitle plus two subtitle lines above the
        # panel titles; at 0.86 the second line landed on them
        gridspec_kw={"hspace": 0.16, "wspace": 0.06,
                     "left": 0.075, "right": 0.985, "top": 0.83, "bottom": 0.04},
    )
    for j, (idx, cap, col) in enumerate(sim_panels):
        axes[0, j].imshow(rgb[idx])
        axes[0, j].set_title(cap, fontsize=10.5, color=col, pad=5)
    for j, (stem, cap, col) in enumerate(real_panels):
        i = names.index(stem)
        axes[1, j].imshow(x[i].permute(1, 2, 0).numpy())
        axes[1, j].set_title(cap, fontsize=10.5, color=col, pad=5)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#b0b7bd")
    axes[0, 0].set_ylabel("SIMULATION\n(training frames)", fontsize=10.5,
                          labelpad=10, color="#2c3e50")
    axes[1, 0].set_ylabel("REAL ROBOT\n(lab captures)", fontsize=10.5,
                          labelpad=10, color="#2c3e50")
    fig.suptitle("What the encoder sees: sim vs real, 64\u00d764 as fed to the network",
                 fontsize=13.5, y=0.975, color="#1b2631")
    fig.text(0.5, 0.922,
             "Camera geometry matched to the measured robot: "
             "height 0.20 m, pitch 2.1\u00b0 down, HFOV 32\u00b0",
             ha="center", fontsize=10, color="#566573")
    fig.text(0.5, 0.885,
             f"Rows matched on apparent slab size; appearance still differs and "
             f"not benignly: latents {maha.mean():.1f}\u00d7 out, probe at chance",
             ha="center", fontsize=10, color="#566573")
    out = ROOT / ARGS.fig_out
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
