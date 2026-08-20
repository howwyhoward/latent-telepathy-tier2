"""WP7 gate: can the encoder's latent carry the hazard bit at DEPLOYMENT
apparent sizes, and does that survive the reality gap?

Isaac-free. The 19 Aug audit showed the shipped chain failed for two stacked
reasons that the old checks could not see:
  1. the slab probe only separates when the hazard fills >25 % of the view;
     the real captures put the pad at 2-21 %, mostly inside the band where the
     probe has no signal IN SIM either;
  2. the real-lab backdrop alone displaces the logit by ~9.5 - 47x the hazard
     signal at deployment coverage - so the sim-fitted direction saturates to
     "hazard" on every real frame (pooled 0.636 == the class base rate).

This gate measures both failure modes directly, with pre-registered
thresholds, so encoder candidates can be compared on one line each:

  GATE A (sim, small-coverage): AUC >= 0.90 for hazard at 2-8 % of the frame
          vs no hazard, on held-out streams. This is the band the robot
          actually sees at 1.5-2.8 m with the 32-deg crop.
  GATE B (real, zero-shot): balanced accuracy >= 0.90 on the labeled handoff
          frames with the SIM-fitted probe. Balanced, not pooled: a constant
          predictor must score 0.5, not the class base rate.
  GATE C (real, refit): leave-one-out balanced accuracy >= 0.90 refitting the
          probe on the real latents - the pre-registered fallback ("re-fit on
          real frames in a session"). Passing C but not B means the bit
          survives the encoder and only the probe direction needs the
          calibration session.

Run (either python env; CPU is fine):

    python spike/gate_probe_coverage.py \
        --jepa_ckpt checkpoints/jepa_realcam20.pt \
        --data /data/howard/isaac/datasets/chokepoint_v3_realcam20.npz \
        --handoff handoff/robomaster_handoff_20260815/pad_captures
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chokepoint.jepa import PixelEncoder  # noqa: E402

# label 1 = red pad (hazard analog) visible; matches analyze_handoff.py
REAL_FRAMES = {
    "ctrl_a": 0, "ctrl_b": 0,
    "red_100": 1, "red_150": 1, "red_200": 1, "red_280": 1,
    "red_l_280": 1, "red_r_280": 1, "rg_split_280": 1,
    "green_200": 0, "green_280": 0,
}

DEPLOY_BAND = (0.02, 0.08)   # pad at 1.5-2.8 m through the 32-deg crop
BANDS = [(0.0, 1e-9), (1e-9, 0.02), (0.02, 0.08), (0.08, 0.25), (0.25, 1.01)]
N_FIT, N_VAL = 40_000, 12_000


def encode(enc, rgb_u8, batch=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, len(rgb_u8), batch):
            x = torch.from_numpy(rgb_u8[i:i + batch]).permute(0, 3, 1, 2).float() / 255
            outs.append(enc(x))
    return torch.cat(outs)


def fit_logistic(Z, y, l2=1e-3, iters=200):
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # pos_weight balances the classes so the fit cannot buy accuracy from the
    # base rate - the exact failure the pooled handoff number hid
    pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    opt = torch.optim.LBFGS([w, b], max_iter=iters)

    def cl():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Z @ w + b, y.float(), pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(cl)
    return w.detach(), b.detach()


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([1.0 * (p > q) + 0.5 * (p == q)
                          for p, q in product(pos, neg)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", default="checkpoints/jepa_realcam20.pt")
    ap.add_argument("--data",
                    default="/data/howard/isaac/datasets/chokepoint_v3_realcam20.npz")
    ap.add_argument("--handoff",
                    default="handoff/robomaster_handoff_20260815/pad_captures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ck = torch.load(ROOT / args.jepa_ckpt, map_location="cpu")
    enc = PixelEncoder(ck["config"]["latent_dim"]).eval()
    enc.load_state_dict(ck["encoder"])
    print(f"encoder: {args.jepa_ckpt} "
          f"(appearance_dr={ck['config'].get('appearance_dr', 0.0)})")

    d = np.load(args.data)
    rgb, seg, valid = d["rgb"], d["seg"], d["valid"]
    streams = int(d["streams"])
    haz = list(d["seg_classes"]).index("hazard")
    cov = (seg == haz).sum(axis=(1, 2)) / (seg.shape[1] * seg.shape[2])

    # split by stream: fit and val never share an episode
    stream_of = np.arange(len(rgb)) % streams
    val_streams = np.random.default_rng(args.seed).choice(streams, 16, replace=False)
    is_val = np.isin(stream_of, val_streams) & valid
    is_fit = ~np.isin(stream_of, val_streams) & valid
    rng = np.random.default_rng(args.seed)
    fi = rng.permutation(np.where(is_fit)[0])[:N_FIT]
    vi = rng.permutation(np.where(is_val)[0])[:N_VAL]

    Zf = encode(enc, rgb[fi])
    Zv = encode(enc, rgb[vi])
    mu, sd = Zf.mean(0), Zf.std(0) + 1e-8
    yf = torch.from_numpy((cov[fi] > 0).astype(np.int64))
    w, b = fit_logistic((Zf - mu) / sd, yf)
    lgv = (((Zv - mu) / sd) @ w + b).numpy()

    print(f"\nsim hazard-visibility probe, held-out streams "
          f"({len(vi)} frames):")
    neg_all = lgv[cov[vi] == 0]
    band_auc = {}
    for lo, hi in BANDS[1:]:
        m = (cov[vi] >= lo) & (cov[vi] < hi)
        a = auc(lgv[m], neg_all)
        band_auc[(lo, hi)] = a
        print(f"  coverage {lo*100:5.1f}-{hi*100:5.1f} %  n={m.sum():5d}  "
              f"AUC vs no-hazard {a:.3f}")

    gate_a = band_auc[DEPLOY_BAND] >= 0.90
    print(f"GATE A (sim, {DEPLOY_BAND[0]*100:.0f}-{DEPLOY_BAND[1]*100:.0f} % "
          f"coverage AUC >= 0.90): {'PASS' if gate_a else 'FAIL'} "
          f"({band_auc[DEPLOY_BAND]:.3f})")

    gate_b = gate_c = None
    hdir = ROOT / args.handoff
    if hdir.exists():
        names = list(REAL_FRAMES)
        y = np.array([REAL_FRAMES[n] for n in names])
        imgs = np.stack([np.load(sorted(hdir.glob(f"{n}_*_64.npy"))[0])
                         for n in names])
        with torch.no_grad():
            Zr = enc(torch.from_numpy(imgs).permute(0, 3, 1, 2))
        lgr = (((Zr - mu) / sd) @ w + b).numpy()
        maha = float((((Zr - mu) / sd) ** 2).mean(dim=1).sqrt().mean())

        tpr = float((lgr[y == 1] > 0).mean())
        tnr = float((lgr[y == 0] <= 0).mean())
        bal = (tpr + tnr) / 2
        print(f"\nreal handoff frames ({len(y)} labeled):")
        print(f"  latent distance from sim cloud    {maha:.2f} "
              f"(in-distribution ~1.0)")
        print(f"  zero-shot TPR {tpr:.3f} / TNR {tnr:.3f} -> balanced "
              f"{bal:.3f}, AUC {auc(lgr[y == 1], lgr[y == 0]):.3f}")
        gate_b = bal >= 0.90
        print(f"GATE B (real zero-shot, balanced acc >= 0.90): "
              f"{'PASS' if gate_b else 'FAIL'} ({bal:.3f})")

        Zs = (Zr - Zr.mean(0)) / (Zr.std(0) + 1e-8)
        hits_p, hits_n = [], []
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            wi, bi = fit_logistic(Zs[m], torch.from_numpy(y[m]), l2=0.1)
            hit = int((Zs[i] @ wi + bi > 0).item() == bool(y[i]))
            (hits_p if y[i] == 1 else hits_n).append(hit)
        bal_loo = (np.mean(hits_p) + np.mean(hits_n)) / 2
        gate_c = bal_loo >= 0.90
        print(f"GATE C (real refit LOO, balanced acc >= 0.90): "
              f"{'PASS' if gate_c else 'FAIL'} ({bal_loo:.3f}, "
              f"n={len(y)} -- treat as indicative until the ~120-frame "
              f"calibration set exists)")

    verdict = {True: "PASS", False: "FAIL", None: "SKIP"}
    print(f"\nSUMMARY  A:{verdict[gate_a]}  B:{verdict[gate_b]}  "
          f"C:{verdict[gate_c]}   ({args.jepa_ckpt})")
    sys.exit(0 if gate_a and (gate_b or gate_c) else 1)


if __name__ == "__main__":
    main()
