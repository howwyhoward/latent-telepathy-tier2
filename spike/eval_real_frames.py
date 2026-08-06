"""WP7 gate measurement — does the sim-trained encoder cross the reality gap?

Isaac-free. Loads the frozen JEPA encoder and the sim-fitted slab probe
(spike/fit_slab_probe.py), runs both on real RoboMaster frames, and reports:

  1. Latent health: per-dim std of real latents vs the sim reference —
     a collapsed or exploded latent fails before any probe is consulted.
  2. Zero-shot probe transfer: slab-side accuracy per session and pooled.
     Session folders are the labels: slab_top_* -> 1, slab_bottom_* -> 0;
     `background*` sessions join the health check but carry no label.

Decision rule (pre-registered): pooled probe accuracy >= 0.9 -> the encoder
transfers, deployment is engineering. Near 0.5 -> domain-randomize the JEPA
data and retrain. In between -> look at the per-session table (lighting
sessions usually explain it).

Frames: rectified pinhole matching the sim camera (82.3 deg FOV, ideally a
SQUARE rectification target). Non-square frames are center-cropped to square,
which narrows the horizontal FOV below the sim's — the report flags it.

    python spike/eval_real_frames.py --frames data/real_frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.jepa import PixelEncoder  # noqa: E402


def load_session(folder: Path, device) -> torch.Tensor:
    """PNG/JPG frames -> (N, 3, 64, 64) float in [0,1], sim preprocessing."""
    import cv2

    frames = []
    files = sorted(
        [*folder.glob("*.png"), *folder.glob("*.jpg"), *folder.glob("*.jpeg")]
    )
    non_square = 0
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w0 = img.shape[:2]
        if h != w0:
            non_square += 1
            s = min(h, w0)
            y0, x0 = (h - s) // 2, (w0 - s) // 2
            img = img[y0 : y0 + s, x0 : x0 + s]
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    if not frames:
        return torch.empty(0, 3, 64, 64, device=device), 0
    x = torch.stack(frames).to(device)
    x = F.interpolate(x, size=(64, 64), mode="area")
    return x, non_square


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=str, default="data/real_frames")
    ap.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
    ap.add_argument("--probe", type=str, default="checkpoints/slab_probe.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ck["encoder"])

    pk = torch.load(args.probe, map_location=device)
    w, b = pk["w"].to(device), pk["b"].to(device)
    sim_std = pk["sim_latent_std"].to(device)
    print(f"probe: sim held-out acc {pk['acc_heldout']:.3f} "
          f"({pk['n_samples']} samples)")

    root = Path(args.frames)
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        sys.exit(f"no session folders under {root}")

    all_z, labeled = [], []
    any_non_square = 0
    print(f"\n{'session':30s} {'n':>5s} {'label':>6s} {'probe_acc':>9s}")
    for s in sessions:
        x, non_square = load_session(s, device)
        any_non_square += non_square
        if len(x) == 0:
            print(f"{s.name:30s} {0:5d}  (no readable frames)")
            continue
        with torch.no_grad():
            z = encoder(x)
        all_z.append(z)
        name = s.name.lower()
        if name.startswith("slab_top"):
            label = 1.0
        elif name.startswith("slab_bottom"):
            label = 0.0
        else:
            print(f"{s.name:30s} {len(x):5d} {'—':>6s} {'(health only)':>9s}")
            continue
        with torch.no_grad():
            pred = ((z @ w + b) > 0).float()
            acc = (pred == label).float().mean().item()
        labeled.append((s.name, len(x), acc))
        print(f"{s.name:30s} {len(x):5d} {label:6.0f} {acc:9.3f}")

    # ---- latent health -------------------------------------------------------
    Z = torch.cat(all_z)
    real_std = Z.std(dim=0)
    ratio = (real_std / (sim_std + 1e-8))
    print(f"\nlatent health ({len(Z)} real frames):")
    print(f"  per-dim std ratio real/sim: median {ratio.median():.2f}  "
          f"min {ratio.min():.2f}  max {ratio.max():.2f}")
    n_dead = int((real_std < 0.05 * sim_std).sum())
    print(f"  dims collapsed on real input (<5% of sim std): {n_dead}/{len(real_std)}")
    if any_non_square:
        print(f"  WARNING: {any_non_square} non-square frames were center-cropped; "
              f"their horizontal FOV is narrower than the sim's 82.3 deg.")

    # ---- the gate number -----------------------------------------------------
    if labeled:
        n_tot = sum(n for _, n, _ in labeled)
        pooled = sum(n * a for _, n, a in labeled) / n_tot
        print(f"\nZERO-SHOT PROBE TRANSFER (pooled, {n_tot} frames): {pooled:.3f}")
        verdict = ("ENCODER TRANSFERS — deployment is engineering"
                   if pooled >= 0.9 else
                   "NO TRANSFER — domain-randomize JEPA data and retrain"
                   if pooled <= 0.6 else
                   "PARTIAL — inspect per-session table (lighting?)")
        print(f"WP7 GATE: {verdict}")
    else:
        print("\nno labeled sessions (slab_top_* / slab_bottom_*) — health check only")


if __name__ == "__main__":
    main()
