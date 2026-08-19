"""Fit a hazard-VISIBLE probe on JEPA-dataset latents (diverse vantages,
class-balanced) and score the 13 real handoff frames — for one encoder.

The scout-vantage slab probe fits its negatives on nearly identical blank-wall
frames, which under the 32-degree camera leaves almost no margin on the absent
class. Dataset frames span all vantages, so this probe's boundary is honest.

    python handoff/fit_dataset_probe.py --tag old \
        --jepa_ckpt checkpoints/jepa_pixels.pt \
        --data isaac-data/datasets/chokepoint_v1.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chokepoint.jepa import PixelEncoder  # noqa: E402
from analyze_handoff_labels import FRAMES, load_real_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min_px", type=int, default=10,
                    help="hazard pixels for a positive label (matches gate margin)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ck["encoder"])

    d = np.load(ROOT / args.data)
    rgb, seg, valid = d["rgb"], d["seg"], d["valid"]
    haz_cls = list(d["seg_classes"]).index("hazard")
    idx = np.where(valid)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    idx = idx[:40000]
    y = (seg[idx] == haz_cls).reshape(len(idx), -1).sum(1) >= args.min_px

    zs = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            x = torch.from_numpy(rgb[idx[i:i+1024]]).permute(0, 3, 1, 2).float().to(device) / 255
            zs.append(encoder(x).cpu())
    Z = torch.cat(zs).numpy()

    n_hold = 4000
    Ztr = torch.from_numpy(Z[:-n_hold]).float()
    ytr = torch.from_numpy(y[:-n_hold]).float()
    Zte = torch.from_numpy(Z[-n_hold:]).float()
    yte = torch.from_numpy(y[-n_hold:]).float()

    w = torch.zeros(Ztr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # balanced BCE so the ~8-16% positive rate doesn't shrink the margin
    pos_w = ((1 - ytr.mean()) / ytr.mean()).item()
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Ztr @ w + b, ytr, pos_weight=torch.tensor(pos_w))
        loss = loss + 1e-4 * w.pow(2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (((Zte @ w + b) > 0).float() == yte).float().mean().item()
    pos_rate = y.mean()
    print(f"[{args.tag}] dataset probe: held-out acc {acc:.3f} "
          f"(pos rate {pos_rate:.3f}, n_train {len(Ztr)})")

    names, x_real, labels = load_real_frames()
    with torch.no_grad():
        z_real = encoder(x_real.to(device)).cpu()
        logits = (z_real @ w + b).numpy()
    hits = n_lab = 0
    print(f"{'frame':16s} {'label':>5s} {'pred':>4s} {'logit':>8s}")
    for i, n in enumerate(names):
        lab = FRAMES[n]
        pred = int(logits[i] > 0)
        mark = ""
        if lab is not None:
            n_lab += 1
            hits += int(pred == lab)
            mark = "  OK" if pred == lab else "  MISS"
        print(f"{n:16s} {'-' if lab is None else lab!s:>5s} {pred:>4d} "
              f"{logits[i]:>8.2f}{mark}")
    print(f"[{args.tag}] ZERO-SHOT (pooled, {n_lab}): {hits/n_lab:.3f}")


if __name__ == "__main__":
    main()
