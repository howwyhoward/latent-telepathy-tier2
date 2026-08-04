"""Does the policy EVER sample the other corridor from the canonical start?

The race has never separated: with a warm-started, corridor-competent trunk,
even a noiseless oracle bit fails to beat silence (race v6). Before tuning
anything else, measure the precondition every one of those runs assumed —
that PPO's Gaussian action noise explores both corridor choices. If the
top-corridor branch is never sampled from the west chamber, no message can
ever be recruited: the gradient that would teach "when the message says X,
go the other way" is computed over trajectories that do not exist.

Sweeps exploration SCALE (log_std) against exploration CORRELATION TIME, the
quantity the arithmetic says actually matters. Reaching the top mouth from
the canonical start needs ~+1.3 of sustained lateral action over the ~30-step
decision window; under iid noise the mean deviation across those steps has
std sigma/sqrt(30) ~ 0.09, i.e. a 14-sigma event at any usable sigma. Under
noise correlated over the window it is a ~1.3/sigma-sigma event, which is
merely rare. Correlated noise keeps the PER-STEP marginal at N(mu, sigma),
so PPO's per-timestep log-probs and importance ratios stay exact.

    python spike/diag_exploration.py --policy runs/nav_pretrain/nav_s1_mouth.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str,
                    default="runs/nav_pretrain/nav_s1_mouth.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--grid", type=str,
                    default="-0.6:0:0:all,0.5:30:0:all,0.5:30:40:all,"
                            "0.5:30:40:y,1.0:30:40:y,1.0:30:60:y,1.5:30:40:y",
                    help="comma-separated log_std:tau:window:dims specs "
                         "(tau in steps, 0=white; window 0 = whole episode; "
                         "dims all|y)")
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
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    policy = AttentionReceiver(encoder, broadcast_dim=0, latent_dim=LATENT_DIM).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device)["policy"])
    policy.eval()
    base_log_std = policy.log_std.detach().clone()
    print(f"policy {args.policy}  trained log_std {base_log_std.tolist()}")

    E = args.num_envs
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)
    empty_msg = torch.zeros(E, 0, 1, device=device)
    empty_mask = torch.zeros(E, 0, device=device)

    print(f"\n{'log_std':>8} {'std':>5} {'tau':>5} {'win':>5} {'dims':>5} "
          f"{'top':>6} {'bottom':>7} {'neither':>8} {'success':>8}")
    for spec in args.grid.split(","):
        ls_s, tau_s, win_s, dims = spec.split(":")
        ls, tau, window = float(ls_s), float(tau_s), int(win_s)
        sigma_base = base_log_std.exp()
        # Explore only on the axis being probed; the rest keep trained noise, so
        # a wide deviation does not also cost the robot its heading and speed.
        boost = torch.ones_like(sigma_base)
        boost[1 if dims == "y" else slice(None)] = float(np.exp(ls)) / sigma_base[
            1 if dims == "y" else slice(None)
        ]
        # AR(1) noise with marginal N(0,1): rho=0 is white, rho->1 is a single
        # deviation held across the episode.
        rho = float(np.exp(-1.0 / tau)) if tau > 0 else 0.0
        eps = torch.randn(E, N_ACTIONS, device=device)
        t = torch.zeros(E, device=device)

        obs, _ = env.reset()
        committed = torch.zeros(E, dtype=torch.long, device=device)  # 0 none,1 top,2 bot
        counts = {"top": 0, "bottom": 0, "neither": 0}
        succ = []
        while len(succ) < args.episodes:
            rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                mean = policy.actor(policy.features(rgb, empty_msg, empty_mask))
            eps = rho * eps + (1.0 - rho**2) ** 0.5 * torch.randn_like(eps)
            hot = (t < window) if window > 0 else torch.ones_like(t, dtype=torch.bool)
            std = sigma_base * torch.where(hot[:, None], boost, 1.0)
            action = mean + std * eps
            t += 1

            obs, _, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            p = env._local_pos(LEARNER)
            fresh = (committed == 0) & (p[:, 0] > -3.0) & (p[:, 1].abs() > 0.5)
            committed = torch.where(
                fresh, torch.where(p[:, 1] > 0, 1, 2).long(), committed
            )
            done = term[LEARNER] | tout[LEARNER]
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    c = int(committed[i].item())
                    counts["top" if c == 1 else ("bottom" if c == 2 else "neither")] += 1
                    succ.append(float(term[LEARNER][i].item()))
                    committed[i] = 0
                eps[idx] = torch.randn_like(eps[idx])  # fresh deviation per episode
                t[idx] = 0
        n = sum(counts.values())
        print(f"{ls:8.2f} {float(np.exp(ls)):5.2f} {tau:5.0f} {window:5d} {dims:>5} "
              f"{counts['top'] / n:6.2f} {counts['bottom'] / n:7.2f} "
              f"{counts['neither'] / n:8.2f} {float(np.mean(succ)):8.2f}")

    print("\n'top' ~0.00 means the corridor branch is never sampled: no message "
          "can be\nrecruited from trajectories that do not exist. The first row "
          "is the regime\nevery race run so far has trained in.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
