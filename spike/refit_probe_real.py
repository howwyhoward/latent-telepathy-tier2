"""WP7 calibration-session refit: fit the hazard probe on REAL latents.

The pre-registered fallback for a failed zero-shot transfer is "re-fit the
probe on real frames in a session" (the encoder stays frozen; the probe is
65 parameters). The 19 Aug audit showed the bit survives the encoder --
leave-one-out on the 11 handoff frames already reads 0.86 balanced -- so this
tool turns a calibration capture into a deployable probe checkpoint.

Inputs are session folders of *_64.npy frames (the standard 640x360 -> crop
[140,0,360,360] -> rotate180 -> 64x64 float32 pipeline from the handoff's
fov_check.py). Labels come from the session folder name:

    hazard_*  -> 1   (red pad visible in the frame)
    clear_*   -> 0   (no red pad: empty arena, green pad, distractors)

Dry run on the 15 Aug handoff (n=11, LOO):

    python spike/refit_probe_real.py \
        --handoff handoff/robomaster_handoff_20260815/pad_captures

Real calibration set (k-fold CV, saves the probe if it clears the gate):

    python spike/refit_probe_real.py \
        --sessions captures/hazard_* captures/clear_* \
        --out checkpoints/slab_probe_real.pt

The saved checkpoint is drop-in for eval_real_frames.py / the robot node:
`w`/`b` act on RAW encoder latents (normalization folded in), and the
`sim_latent_mean/std` keys hold the CALIBRATION-set stats so the existing
latent-health check measures drift from the calibration session.
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

HANDOFF_LABELS = {
    "ctrl_a": 0, "ctrl_b": 0,
    "red_100": 1, "red_150": 1, "red_200": 1, "red_280": 1,
    "red_l_280": 1, "red_r_280": 1, "rg_split_280": 1,
    "green_200": 0, "green_280": 0,
}
GATE_BALANCED = 0.90


def load_frames(args):
    imgs, y, srcs = [], [], []
    if args.handoff:
        hdir = ROOT / args.handoff
        for stem, lab in HANDOFF_LABELS.items():
            p = sorted(hdir.glob(f"{stem}_*_64.npy"))[0]
            imgs.append(np.load(p)); y.append(lab); srcs.append(stem)
    for sess in args.sessions:
        sp = Path(sess)
        name = sp.name
        if name.startswith("hazard"):
            lab = 1
        elif name.startswith("clear"):
            lab = 0
        else:
            sys.exit(f"session folder must start with hazard_/clear_: {name}")
        files = sorted(sp.glob("*_64.npy")) or sorted(sp.glob("*.npy"))
        if not files:
            sys.exit(f"no .npy frames in {sp}")
        for p in files:
            imgs.append(np.load(p)); y.append(lab); srcs.append(f"{name}/{p.name}")
    x = np.stack(imgs).astype(np.float32)
    assert x.shape[1:] == (64, 64, 3), f"bad frame shape {x.shape}"
    return torch.from_numpy(x).permute(0, 3, 1, 2), np.array(y), srcs


def fit(Z, y, l2=0.1):
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pw = torch.tensor([max((y == 0).sum(), 1) / max((y == 1).sum(), 1)])
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def cl():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Z @ w + b, torch.from_numpy(y).float(), pos_weight=pw
        ) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(cl)
    return w.detach(), b.detach()


def balanced_acc(lg, y):
    tpr = float((lg[y == 1] > 0).mean()) if (y == 1).any() else np.nan
    tnr = float((lg[y == 0] <= 0).mean()) if (y == 0).any() else np.nan
    return (tpr + tnr) / 2, tpr, tnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", default="checkpoints/jepa_realcam20.pt")
    ap.add_argument("--handoff", default=None,
                    help="pad_captures dir with the built-in 11-frame labels")
    ap.add_argument("--sessions", nargs="*", default=[],
                    help="session folders named hazard_* / clear_*")
    ap.add_argument("--out", default=None,
                    help="save the final probe here if CV clears the gate")
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.handoff and not args.sessions:
        sys.exit("give --handoff and/or --sessions")
    torch.manual_seed(args.seed)

    ck = torch.load(ROOT / args.jepa_ckpt, map_location="cpu")
    enc = PixelEncoder(ck["config"]["latent_dim"]).eval()
    enc.load_state_dict(ck["encoder"])

    x, y, srcs = load_frames(args)
    with torch.no_grad():
        Z = enc(x)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"{len(y)} frames ({n_pos} hazard / {n_neg} clear) through "
          f"{args.jepa_ckpt}")
    if min(n_pos, n_neg) < 2:
        sys.exit("need at least 2 frames per class")

    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd

    # cross-validation: LOO when tiny, stratified k-fold otherwise
    rng = np.random.default_rng(args.seed)
    if len(y) <= 20:
        folds = [np.array([i]) for i in range(len(y))]
        print(f"CV: leave-one-out ({len(y)} folds)")
    else:
        folds = [[] for _ in range(args.folds)]
        for cls in (0, 1):
            idx = rng.permutation(np.where(y == cls)[0])
            for k, i in enumerate(idx):
                folds[k % args.folds].append(i)
        folds = [np.array(sorted(f)) for f in folds]
        print(f"CV: stratified {args.folds}-fold")

    lg_cv = np.zeros(len(y))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        w, b = fit(Zs[tr], y[tr], l2=args.l2)
        lg_cv[te] = (Zs[te] @ w + b).numpy()
    bal, tpr, tnr = balanced_acc(lg_cv, y)
    a = float(np.mean([1.0 * (p > q) + 0.5 * (p == q) for p, q
                       in product(lg_cv[y == 1], lg_cv[y == 0])]))
    print(f"CV balanced acc {bal:.3f} (TPR {tpr:.3f} / TNR {tnr:.3f}), "
          f"AUC {a:.3f}")
    miss = [s for s, l, ok in zip(srcs, lg_cv, (lg_cv > 0) == y) if not ok]
    if miss:
        print(f"misclassified: {', '.join(miss[:12])}"
              + (" ..." if len(miss) > 12 else ""))

    ok = bal >= GATE_BALANCED
    print(f"GATE (CV balanced acc >= {GATE_BALANCED}): "
          f"{'PASS' if ok else 'FAIL'}")

    if args.out:
        if not ok:
            print(f"not saving {args.out}: gate failed")
            sys.exit(1)
        w, b = fit(Zs, y, l2=args.l2)
        in_bal, *_ = balanced_acc((Zs @ w + b).numpy(), y)
        # fold the z-score into w/b so deployment applies RAW latents,
        # exactly like the sim-fitted probe it replaces
        w_raw = w / sd
        b_raw = b - (mu / sd) @ w
        torch.save({
            "w": w_raw, "b": b_raw,
            "acc_train": in_bal, "acc_heldout": bal,
            "n_samples": len(y),
            "sim_latent_mean": Z.mean(0), "sim_latent_std": Z.std(0),
            "jepa_ckpt": args.jepa_ckpt,
            "fit_on": "real_frames",
            "sources": srcs,
        }, ROOT / args.out)
        print(f"saved -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
