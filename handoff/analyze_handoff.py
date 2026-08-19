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
    hits, n_lab = 0, 0
    for i, n in enumerate(names):
        lab = labels[i]
        ls = "-" if lab is None else str(lab)
        ok = ""
        if lab is not None:
            n_lab += 1
            hit = int(pred[i].item() == lab)
            hits += hit
            ok = "  OK" if hit else "  MISS"
        a = act[i].tolist()
        print(f"{n:16s} {ls:>5s} {pred[i].item():>5d} {logit[i].item():>8.2f} "
              f"[{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]{ok}")

    print(f"\nZERO-SHOT PROBE TRANSFER (pooled, {n_lab} labeled frames): "
          f"{hits/n_lab:.3f}")
    print("pre-registered rule: >=0.9 transfers | <=0.6 no transfer | else partial")

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
    vis_idx = np.where(valid & (haz_px > 40))[0][:4]
    non_idx = np.where(valid & (haz_px == 0))[0][:4]
    sim_x = torch.from_numpy(
        rgb[np.concatenate([vis_idx, non_idx])]).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        sim_z = encoder(sim_x)
        sim_logit = sim_z @ w + b.squeeze()
        sim_maha = (((sim_z - sim_mu) / (sim_std + 1e-8)) ** 2).mean(dim=1).sqrt()
    print(f"\nsim sanity: hazard-visible logits {sim_logit[:4].tolist()}")
    print(f"            hazard-absent  logits {sim_logit[4:].tolist()}")
    print(f"            sim frame distance from sim cloud: {sim_maha.mean():.2f}")

    # side-by-side figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAZ = "#c0392b"
    SAFE = "#5d6d7e"
    # sim exemplars matched to the real columns: hazard near, hazard far, none
    order = vis_idx[np.argsort(-haz_px[vis_idx])]
    sim_panels = [
        (order[0], "hazard close", HAZ),
        (order[len(order) // 2], "hazard mid", HAZ),
        (order[-1], "hazard far", HAZ),
        (non_idx[0], "no hazard", SAFE),
    ]
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
             "Remaining differences are appearance only: wall tint, "
             "carpet texture, lighting",
             ha="center", fontsize=10, color="#566573")
    out = ROOT / ARGS.fig_out
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
