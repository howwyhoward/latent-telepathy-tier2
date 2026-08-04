"""How does the v3 z_t policy avoid the slab WITHOUT the message?

Runs the saved policy with the message channel masked, records navigator
trajectories, and plots them split by slab side over the scene geometry.
Distinguishes the two hypotheses:

  peek-and-switch : trajectories enter a corridor, reverse at the slab,
                    exit and take the other one (long ep_len on one side)
  early visibility: trajectories fork correctly from the start (occlusion
                    of the slab from the navigator's approach is broken)

    python spike/diag_race_traj.py --policy runs/race_v3/z_t_s1.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="runs/race_v3/z_t_s1.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--mask_message", type=int, default=1)
parser.add_argument("--out", type=str, default="spike/out/race_traj.png")
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    obs, _ = env.reset()

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    blob = torch.load(args.policy, map_location=device)
    bus = LatentBroadcast(encoder, comm_radius=12.0,
                          broadcast_dim=LATENT_DIM, anchored=True)
    policy = AttentionReceiver(
        encoder, broadcast_dim=bus.wire_dim, latent_dim=LATENT_DIM
    ).to(device)
    policy.load_state_dict(blob["policy"])
    policy.eval()

    zero_scout = torch.zeros(args.num_envs, N_ACTIONS, device=device)

    trajs = [[] for _ in range(args.num_envs)]
    episodes = []  # (traj list of xy, slab_top, success, ep_len, hazard)
    acc_h = torch.zeros(args.num_envs, device=device)
    acc_len = torch.zeros(args.num_envs, device=device)
    slab = env._slab_top.clone()

    while len(episodes) < args.episodes:
        rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
        msgs, mask = bus.deliver(env)[LEARNER]
        mask = mask.float()
        if args.mask_message:
            mask = torch.zeros_like(mask)
        with torch.no_grad():
            h = policy.features(rgb, msgs, mask)
            action = policy.actor(h)
        obs, rew, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        pos = env._local_pos(LEARNER).cpu().numpy()
        acc_h += env._in_hazard(LEARNER).float()
        acc_len += 1
        done = term[LEARNER] | tout[LEARNER]
        for i in range(args.num_envs):
            trajs[i].append(pos[i].copy())
            if done[i]:
                episodes.append((
                    np.array(trajs[i]),
                    bool(slab[i].item()),
                    float(term[LEARNER][i].item()),
                    acc_len[i].item(),
                    acc_h[i].item(),
                ))
                trajs[i] = []
                acc_h[i] = 0.0
                acc_len[i] = 0.0
        slab = env._slab_top.clone()

    for side, name in [(True, "slab TOP"), (False, "slab BOTTOM")]:
        eps = [e for e in episodes if e[1] == side]
        if not eps:
            continue
        print(f"{name}: n={len(eps)}  success {np.mean([e[2] for e in eps]):.2f}  "
              f"ep_len {np.mean([e[3] for e in eps]):6.1f}  "
              f"hazard {np.mean([e[4] for e in eps]):5.1f}")

    geo = env._geo
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, side, title in [(axes[0], True, "slab TOP"), (axes[1], False, "slab BOTTOM")]:
        grid = env._grid
        n = grid.shape[0]
        ax.imshow(
            (grid == 1).T, origin="lower", cmap="Greys", alpha=0.6,
            extent=[-n * cfg.cell / 2, n * cfg.cell / 2,
                    -n * cfg.cell / 2, n * cfg.cell / 2],
        )
        aabb = geo.hazard_aabb_top if side else geo.hazard_aabb_bottom
        ax.add_patch(plt.Rectangle((aabb[0], aabb[2]), aabb[1] - aabb[0],
                                   aabb[3] - aabb[2], color="red", alpha=0.4))
        for traj, s_top, succ, _, _ in episodes:
            if s_top != side:
                continue
            ax.plot(traj[:, 0], traj[:, 1], lw=1.2,
                    color="tab:green" if succ else "tab:orange", alpha=0.7)
        gx, gy = geo.goals[LEARNER]
        ax.plot(gx, gy, "b*", ms=15)
        ax.set_title(f"{title} (green=success, orange=timeout)")
        ax.set_aspect("equal")
    fig.suptitle(f"masked={bool(args.mask_message)}  policy={args.policy}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
