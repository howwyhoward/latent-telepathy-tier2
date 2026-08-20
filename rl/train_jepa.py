"""JEPA Stage-A training on pixels: self-supervised, then freeze (Tier 2 M8).

Isaac-free port of Tier 1's rl/train_jepa.py — same pipeline:

  1. Load random-policy transitions collected by spike/collect_jepa_data.py.
  2. Train online encoder E + predictor P (BYOL/I-JEPA: EMA target,
     stop-gradient, normalized-MSE) + VICReg safety net + the segmentation
     reconstruction auxiliary (hazard/goal/agent upweighted, decoded from
     BOTH z_t and z_pred so the predictor carries the content too).
  3. Collapse monitor every eval interval (per-dim std + effective rank).
  4. Pre-registered probe gates on frozen latents (val split by env stream,
     no episode leakage):
       - hazard-visible linear probe  : acc must beat the majority baseline
       - goal-visible   linear probe  : acc must beat the majority baseline
       - wall-pixel-count regression  : R^2 > 0.5
       - latent health                : min_std >= 1e-2, eff_rank > 0.3*dim
  5. Freeze + checkpoint the encoder for the receiver/message bus.

Run:  python rl/train_jepa.py --data /data/howard/isaac/datasets/chokepoint_v1.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chokepoint.jepa import (  # noqa: E402
    JEPA,
    SEG_CLASSES,
    SEG_RES,
    SegDecoder,
    covariance_loss,
    jepa_loss,
    latent_stats,
    make_class_weights,
    reconstruction_loss,
    variance_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="/data/howard/isaac/datasets/chokepoint_v1.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.99)
    # VICReg safety net engaged by default (collapsed without it in Tier 1)
    p.add_argument("--var-coef", type=float, default=1.0)
    p.add_argument("--cov-coef", type=float, default=0.04)
    # Reconstruction auxiliary ON by default: Tier 1 needed it for the rare
    # goal tokens, and hazard pixels here are just as rare (M10c finding).
    p.add_argument("--recon-coef", type=float, default=1.0)
    p.add_argument("--recon-class-weight", type=float, default=10.0)
    p.add_argument("--val-streams", type=int, default=16,
                   help="streams held out for validation (of num_envs*2)")
    # WP7 finding: the real-lab background alone displaces the slab-probe
    # logit by ~9.5 (47x the hazard signal at deployment distances) because
    # training saw exactly one appearance. Randomizing backdrop appearance
    # per sample, with recon targets unchanged, forces the invariance.
    p.add_argument("--appearance-dr", type=float, default=0.0,
                   help="seg-guided appearance randomization strength "
                        "(0 = off, 1 = full)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--min-std", type=float, default=1e-2)
    p.add_argument("--out", type=str, default="checkpoints/jepa_pixels.pt")
    p.add_argument("--log-csv", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def downsample_seg(seg: torch.Tensor) -> torch.Tensor:
    """(B, 64, 64) -> (B, SEG_RES, SEG_RES) by max-pooling class indices.

    Max keeps rare high-index classes (hazard/goal/agent) alive at 16x16;
    plain striding can drop a 2-pixel hazard sliver entirely.
    """
    k = seg.shape[-1] // SEG_RES
    return torch.nn.functional.max_pool2d(seg.float().unsqueeze(1), k).squeeze(1).long()


# -- appearance randomization (WP7 sim-to-real fix) --------------------------

BACKDROP_CLASSES = (0, 1)  # background (incl. floor) and wall


def draw_appearance(bsz: int, strength: float, device) -> dict:
    """One appearance draw per transition pair.

    Lighting and surface appearance do not change between consecutive frames,
    so the same draw must be applied to x_t and x_{t+1}; only the seg masks
    differ with viewpoint.
    """
    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=device)

    s = strength
    k = len(BACKDROP_CLASSES)
    # independent per-channel tints for floor/background vs wall: the lab
    # has dark carpet under pink-tinted walls, the sim has uniform grey
    tint_scale = 1.0 + u(-0.35 * s, 0.35 * s, bsz, k, 3)
    tint_off = u(-0.12 * s, 0.12 * s, bsz, k, 3)
    # Hue guard: never let the backdrop go red. A red-tinted wall teaches the
    # encoder that red backdrop is plausible -- exactly the hazard confusion
    # deployment cannot afford (the unguarded v1/v2 draws produced brick-red
    # walls and scored worse on the real frames than no DR at all).
    grey = 0.55
    ch = grey * tint_scale + tint_off  # backdrop mid-grey after tint, (b,k,3)
    excess = (ch[..., 0] - torch.maximum(ch[..., 1], ch[..., 2]) - 0.05
              ).clamp(min=0)
    tint_off[..., 0] -= excess
    return {
        "tint_scale": tint_scale,
        "tint_off": tint_off,
        # per-class two-band texture: the carpet is pixel-level speckle at
        # 64x64 while walls vary smoothly, so each backdrop class gets its own
        # low-frequency field plus an independent high-frequency component
        # (the first DR attempt shared one smooth field and scored WORSE on
        # the real frames than no DR at all)
        "tex_amp_lo": u(0.0, 0.18 * s, bsz, k, 1, 1, 1),
        "tex_lo": torch.randn(bsz, k, 1, 8, 8, device=device),
        "tex_amp_hi": u(0.0, 0.12 * s, bsz, k, 1, 1, 1),
        "tex_hi": torch.randn(bsz, k, 1, 64, 64, device=device),
        # global photometrics: exposure, gamma, sensor noise
        "gamma": torch.exp(u(-0.4 * s, 0.4 * s, bsz, 1, 1, 1)),
        "gain": 1.0 + u(-0.3 * s, 0.3 * s, bsz, 1, 1, 1),
        "noise_std": u(0.0, 0.03 * s, bsz, 1, 1, 1),
    }


def apply_appearance(x: torch.Tensor, seg_full: torch.Tensor, d: dict) -> torch.Tensor:
    """x (B,3,64,64) in [0,1]; seg_full (B,64,64) class indices at full res.

    Hazard/goal/agent pixels keep their identity (only global photometrics
    touch them): the red pad's hue transfers, it is the backdrop that does
    not, so that is what gets randomized.
    """
    out = x
    for k, ci in enumerate(BACKDROP_CLASSES):
        lo = torch.nn.functional.interpolate(
            d["tex_lo"][:, k], size=x.shape[-2:], mode="bilinear",
            align_corners=False)
        tex = 1.0 + d["tex_amp_lo"][:, k] * lo + d["tex_amp_hi"][:, k] * d["tex_hi"][:, k]
        m = (seg_full == ci).unsqueeze(1).float()
        scale = d["tint_scale"][:, k].view(-1, 3, 1, 1)
        off = d["tint_off"][:, k].view(-1, 3, 1, 1)
        out = out * (1 - m) + (out * scale * tex + off) * m
    out = d["gain"] * out.clamp(0.0, 1.0) ** d["gamma"]
    out = out + d["noise_std"] * torch.randn_like(out)
    return out.clamp(0.0, 1.0)


# -- probes (Tier 1's run_probe, targets adapted to pixels) -----------------


def _encode_all(encoder, rgb_u8: np.ndarray, device, batch=2048) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, len(rgb_u8), batch):
            x = torch.from_numpy(rgb_u8[i : i + batch]).to(device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            outs.append(encoder(x).cpu())
    return torch.cat(outs, dim=0)


def _train_probe(x, y, *, classification, hidden=0, epochs=300, lr=1e-2):
    in_dim = x.shape[1]
    if hidden:
        head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )
    else:
        head = torch.nn.Linear(in_dim, 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss()
    y = y.float().view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(head(x), y).backward()
        opt.step()
    return head


def run_probes(encoder, train_rgb, train_seg, val_rgb, val_seg, device):
    """Visibility probes for the thesis-critical classes + wall-count R^2."""
    zx = _encode_all(encoder, train_rgb, device)
    vx = _encode_all(encoder, val_rgb, device)
    mu, sd = zx.mean(0, keepdim=True), zx.std(0, keepdim=True) + 1e-6
    zx, vx = (zx - mu) / sd, (vx - mu) / sd

    metrics = {}
    for cls in ("hazard", "goal"):
        ci = SEG_CLASSES.index(cls)
        y_tr = torch.from_numpy((train_seg == ci).any(axis=(1, 2)).astype(np.float32))
        y_va = (val_seg == ci).any(axis=(1, 2)).astype(np.float32)
        majority = float(max(y_va.mean(), 1 - y_va.mean()))

        def acc(hidden):
            clf = _train_probe(zx, y_tr, classification=True, hidden=hidden)
            with torch.no_grad():
                pred = (torch.sigmoid(clf(vx)).view(-1) > 0.5).float().numpy()
            return float((pred == y_va).mean())

        metrics[f"{cls}_acc"] = acc(0)
        metrics[f"{cls}_acc_mlp"] = acc(128)
        metrics[f"{cls}_majority"] = majority

    wall_i = SEG_CLASSES.index("wall")
    wc_tr = torch.from_numpy((train_seg == wall_i).sum(axis=(1, 2)).astype(np.float32))
    wc_va = (val_seg == wall_i).sum(axis=(1, 2)).astype(np.float32)
    wc_mu, wc_sd = float(wc_tr.mean()), float(wc_tr.std()) + 1e-6
    reg = _train_probe(zx, (wc_tr - wc_mu) / wc_sd, classification=False)
    with torch.no_grad():
        wpred = reg(vx).view(-1).numpy() * wc_sd + wc_mu
    ss_res = float(((wc_va - wpred) ** 2).sum())
    ss_tot = float(((wc_va - wc_va.mean()) ** 2).sum()) + 1e-12
    metrics["wall_r2"] = 1.0 - ss_res / ss_tot
    return metrics


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    data = np.load(args.data)
    rgb, seg = data["rgb"], data["seg"]
    action, valid = data["action"], data["valid"]
    streams = int(data["streams"])
    n_steps = len(rgb) // streams
    print(f"loaded {len(rgb)} frames ({n_steps} steps x {streams} streams) "
          f"in {time.time() - t0:.1f}s")

    # split by stream so train and val never share an episode
    stream_of = np.arange(len(rgb)) % streams
    val_streams = np.random.default_rng(args.seed).choice(
        streams, size=args.val_streams, replace=False
    )
    is_val = np.isin(stream_of, val_streams)

    # transition pairs: successor of i is i + streams (same stream, next step)
    idx_all = np.arange(len(rgb) - streams)
    pairs = idx_all[valid[idx_all]]
    train_pairs = pairs[~is_val[pairs]]
    val_pairs = pairs[is_val[pairs]][:4096]
    print(f"transitions: train {len(train_pairs)}  val {len(val_pairs)}")

    model = JEPA().to(device)
    decoder = SegDecoder().to(device)
    class_weights = make_class_weights(args.recon_class_weight, device=device)
    params = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(decoder.parameters())
    )
    opt = torch.optim.Adam(params, lr=args.lr)

    def fetch(idx: np.ndarray, augment: bool = False):
        """Gather a transition batch onto the GPU as float tensors."""
        x_t = torch.from_numpy(rgb[idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x_n = torch.from_numpy(rgb[idx + streams]).to(device).permute(0, 3, 1, 2).float() / 255.0
        a_t = torch.from_numpy(action[idx]).to(device)
        sf_t = torch.from_numpy(seg[idx]).to(device)
        sf_n = torch.from_numpy(seg[idx + streams]).to(device)
        if augment and args.appearance_dr > 0:
            # recon targets stay the un-augmented seg: semantics must survive
            # every appearance, which is the whole point
            d = draw_appearance(len(idx), args.appearance_dr, device)
            x_t = apply_appearance(x_t, sf_t, d)
            x_n = apply_appearance(x_n, sf_n, d)
        return x_t, a_t, x_n, downsample_seg(sf_t), downsample_seg(sf_n)

    vx_t, va_t, vx_n, _, _ = fetch(val_pairs)

    csv_file = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w")
        csv_file.write("step,inv_train,inv_val,eff_rank\n")

    print("training...")
    rng = np.random.default_rng(args.seed)
    global_step = 0
    last_inv = float("nan")
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_pairs))
        model.train()
        for start in range(0, len(order), args.batch_size):
            idx = train_pairs[order[start : start + args.batch_size]]
            x_t, a_t, x_n, s_t, s_n = fetch(idx, augment=True)

            z_pred, z_target, z_t = model(x_t, a_t, x_n)
            inv = jepa_loss(z_pred, z_target)
            var = variance_loss(z_t) + variance_loss(z_pred)
            cov = covariance_loss(z_t) + covariance_loss(z_pred)
            # decode current seg from z_t AND next seg from z_pred (shared
            # decoder): the second term forces the PREDICTOR to carry
            # hazard/goal content forward (Tier 1's goal-blindness fix)
            rec = reconstruction_loss(decoder(z_t), s_t, class_weights) + \
                reconstruction_loss(decoder(z_pred), s_n, class_weights)
            loss = inv + args.var_coef * var + args.cov_coef * cov + args.recon_coef * rec

            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target(tau=args.tau)
            global_step += 1
            last_inv = inv.item()

            if global_step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    zp_v, zt_v, zv = model(vx_t, va_t, vx_n)
                    val_inv = jepa_loss(zp_v, zt_v).item()
                mean_std, min_std, eff_rank = latent_stats(zv)
                flag = "  <-- COLLAPSE?" if min_std < args.min_std else ""
                print(
                    f"epoch {epoch} step {global_step:>6d}  loss {loss.item():.4f} "
                    f"(inv {inv.item():.4f} val_inv {val_inv:.4f} var {var.item():.3f} "
                    f"cov {cov.item():.3f} rec {rec.item():.3f})  "
                    f"mean_std {mean_std:.3f}  eff_rank {eff_rank:4.1f}/"
                    f"{model.encoder.latent_dim}{flag}",
                    flush=True,
                )
                if csv_file:
                    csv_file.write(f"{global_step},{inv.item():.6f},{val_inv:.6f},{eff_rank:.3f}\n")
                    csv_file.flush()
                model.train()

    if csv_file:
        csv_file.close()

    # -- pre-registered probe gates on frozen latents ----------------------
    print("\nrunning probes on frozen encoder...")
    model.encoder.eval()
    tr_mask = ~is_val
    tr_idx = np.where(tr_mask)[0][:40_000]
    va_idx = np.where(is_val)[0][:8_000]
    metrics = run_probes(
        model.encoder, rgb[tr_idx], seg[tr_idx], rgb[va_idx], seg[va_idx], device
    )
    for cls in ("hazard", "goal"):
        print(
            f"  {cls}-visible acc: linear {metrics[f'{cls}_acc']:.3f}  "
            f"mlp {metrics[f'{cls}_acc_mlp']:.3f}  "
            f"(majority {metrics[f'{cls}_majority']:.3f})"
        )
    print(f"  wall-count R^2  {metrics['wall_r2']:.3f}")

    zv = _encode_all(model.encoder, rgb[va_idx][:4096], device)
    mean_std, min_std, eff_rank = latent_stats(zv)
    print(f"  final latents: mean_std {mean_std:.3f}  min_std {min_std:.3f}  "
          f"eff_rank {eff_rank:.1f}/{model.encoder.latent_dim}")

    gates = {
        "hazard_probe": metrics["hazard_acc"] > metrics["hazard_majority"],
        "goal_probe": metrics["goal_acc"] > metrics["goal_majority"],
        "wall_r2": metrics["wall_r2"] > 0.5,
        "latent_health": min_std >= args.min_std
        and eff_rank > model.encoder.latent_dim * 0.3,
    }
    for name, ok in gates.items():
        print(f"  gate {name}: {'PASS' if ok else 'FAIL'}")
    overall = all(gates.values())
    print(f"\nM8 STAGE-A {'PASS' if overall else 'FAIL'}  (inv {last_inv:.4f})")

    # -- freeze + checkpoint ------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "config": {
                "latent_dim": model.encoder.latent_dim,
                "seg_classes": list(SEG_CLASSES),
                "recon_coef": args.recon_coef,
                "appearance_dr": args.appearance_dr,
            },
            "probe_metrics": metrics,
            "gates": gates,
            "final_inv_loss": last_inv,
        },
        out,
    )
    print(f"saved frozen encoder -> {out}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
