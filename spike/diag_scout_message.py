"""Diagnose the race null result. Two questions, tested independently:

  A. INFORMATION: does the parked scout's camera (hence its frozen latent
     z_t) actually disambiguate the slab side? Linear probe slab_top vs
     z_scout over many resets, plus the raw-pixel difference between sides.
     If this fails, no trainer change can rescue z_t — the message is blank.

  B. RECRUITMENT: does the trained z_t policy read the message at all?
     Run the saved policy with real messages vs a zeroed mask. If success /
     hazard / corridor choice are identical, the channel was never recruited
     and the failure is optimization economics, not information.

    python spike/diag_scout_message.py --policy runs/race/z_t_s1.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="runs/race/z_t_s1.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--probe_resets", type=int, default=16,
                    help="reset rounds for part A (samples = rounds x envs)")
parser.add_argument("--eval_episodes", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import traceback


def _die_loudly(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _die_loudly

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.message_bus import LatentBroadcast  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device
    env.reset()

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    zero = {a: torch.zeros(args.num_envs, N_ACTIONS, device=device)
            for a in (LEARNER, BEACON)}

    # ---------------- part A: information content of the scout view ---------
    print("\n=== A. scout-view information ===")
    frames, labels, latents = [], [], []
    for _ in range(args.probe_resets):
        env._reset_idx(torch.arange(args.num_envs, device=device))
        # let physics + renderer settle at the parked pose
        for _ in range(3):
            obs, *_ = env.step(zero)
        rgb = obs[BEACON]                       # (B, 64, 64, 3) in [0,1]
        z = encoder(rgb.permute(0, 3, 1, 2).contiguous())
        frames.append(rgb.cpu().numpy())
        latents.append(z.cpu().numpy())
        labels.append(env._slab_top.cpu().numpy())
    X = np.concatenate(latents)
    F = np.concatenate(frames)
    y = np.concatenate(labels).astype(int)
    print(f"samples: {len(y)}  slab_top rate: {y.mean():.2f}")

    top_mean = F[y == 1].mean(0)
    bot_mean = F[y == 0].mean(0)
    pix_diff = np.abs(top_mean - bot_mean)
    print(f"pixel |mean_top - mean_bot|: mean {pix_diff.mean():.5f}  "
          f"max {pix_diff.max():.5f}  pixels>0.05: {(pix_diff.max(-1) > 0.05).sum()}")

    # simple logistic probe, 75/25 split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr = int(0.75 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        logit = Xt[tr] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[tr])
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc_tr = (((Xt[tr] @ w + b) > 0).float() == yt[tr]).float().mean()
        acc_te = (((Xt[te] @ w + b) > 0).float() == yt[te]).float().mean()
    print(f"linear probe slab_top from z_scout: train {acc_tr:.3f}  TEST {acc_te:.3f}")
    print("VERDICT A:", "message CARRIES slab side" if acc_te > 0.9
          else "message does NOT separate slab side -> fix scout view first")

    # ---------------- part B: does the trained policy read it? --------------
    print("\n=== B. trained z_t policy, message ablation ===")
    blob = torch.load(args.policy, map_location=device)
    bus = LatentBroadcast(encoder, comm_radius=12.0,
                          broadcast_dim=LATENT_DIM, anchored=True)
    policy = AttentionReceiver(
        encoder, broadcast_dim=bus.wire_dim, latent_dim=LATENT_DIM
    ).to(device)
    policy.load_state_dict(blob["policy"])
    policy.eval()

    def run_eval(use_message: bool):
        obs, _ = env.reset()
        done_count, succ, haz, went_top = 0, [], [], []
        acc_h = torch.zeros(args.num_envs, device=device)
        max_y = torch.full((args.num_envs,), -1e9, device=device)
        while done_count < args.eval_episodes:
            rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
            msgs, mask = bus.deliver(env)[LEARNER]
            mask = mask.float()
            if not use_message:
                mask = torch.zeros_like(mask)
            with torch.no_grad():
                h = policy.features(rgb, msgs, mask)
                action = policy.actor(h)  # deterministic: mean action
            obs, rew, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero[BEACON]}
            )
            acc_h += env._in_hazard(LEARNER).float()
            max_y = torch.maximum(max_y, env._local_pos(LEARNER)[:, 1])
            done = term[LEARNER] | tout[LEARNER]
            if done.any():
                for i in done.nonzero(as_tuple=True)[0].tolist():
                    succ.append(float(term[LEARNER][i].item()))
                    haz.append(acc_h[i].item())
                    went_top.append(float(max_y[i].item() > 0.0))
                    acc_h[i] = 0.0
                    max_y[i] = -1e9
                    done_count += 1
        return np.mean(succ), np.mean(haz), np.mean(went_top), len(succ)

    s1, h1, t1, n1 = run_eval(use_message=True)
    s0, h0, t0, n0 = run_eval(use_message=False)
    print(f"with message   : success {s1:.3f}  hazard {h1:5.1f}  went_top {t1:.2f}  (n={n1})")
    print(f"masked message : success {s0:.3f}  hazard {h0:5.1f}  went_top {t0:.2f}  (n={n0})")
    print("VERDICT B:", "policy USES the message" if abs(s1 - s0) > 0.05 or abs(h1 - h0) > 3
          else "policy IGNORES the message -> optimization/economics failure")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
