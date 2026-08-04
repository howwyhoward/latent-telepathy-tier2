"""Does the message change which corridor the trained policy actually takes?

diag_msg_sensitivity shows the v7 oracle's lateral mean swings by 1.6 when the
slab bit flips -- yet slab-bottom success is 0.01. Exactly one of these is true
and they need opposite fixes:

  (a) the policy DOES route on the message and then fails to traverse the
      corridor it chose  -> an execution problem in the trunk, and
  (b) the closed-form sensitivity does not survive at real operating points, or
      the routing is the wrong way round -> still a decision problem.

Runs the policy deterministically (action = mean, no exploration) from the
canonical start with the TRUE bit and with the bit FLIPPED, and cross-tabulates
slab side against the corridor actually entered. Flipping is the causal test:
if the corridor follows the bit rather than the world, the message is steering.

    python spike/diag_route_choice.py --policy runs/race_v7/oracle.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="runs/race_v7/oracle.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--comm_radius", type=float, default=12.0)
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
from chokepoint.message_bus import OracleBroadcast  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"
ANCHOR_DIMS = 2


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

    bus = OracleBroadcast(comm_radius=args.comm_radius, broadcast_dim=LATENT_DIM,
                          anchored=True)
    wire = ANCHOR_DIMS + LATENT_DIM
    policy = AttentionReceiver(encoder, broadcast_dim=wire, latent_dim=LATENT_DIM).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device)["policy"])
    policy.eval()

    E = args.num_envs
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)

    for flip in (False, True):
        obs_dict, _ = env.reset()
        committed = torch.zeros(E, dtype=torch.long, device=device)  # 0 none,1 top,2 bot
        # cross-tab[slab_top][corridor] and success by slab side
        tab = {(s, c): 0 for s in (0, 1) for c in (0, 1, 2)}
        succ = {0: [], 1: []}
        ep_slab = env._slab_top.clone()
        n = 0
        while n < args.episodes:
            rgb = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
            msg, mask = bus.deliver(env)[LEARNER]
            if flip:
                msg[:, :, ANCHOR_DIMS] *= -1.0
            with torch.no_grad():
                action = policy.actor(policy.features(rgb, msg, mask.float()))

            obs_dict, _, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            p = env._local_pos(LEARNER)
            fresh = (committed == 0) & (p[:, 0] > -3.0) & (p[:, 1].abs() > 0.5)
            committed = torch.where(
                fresh, torch.where(p[:, 1] > 0, 1, 2).long(), committed
            )
            done = term[LEARNER] | tout[LEARNER]
            if done.any():
                for i in done.nonzero(as_tuple=True)[0].tolist():
                    s = int(ep_slab[i].item())
                    tab[(s, int(committed[i].item()))] += 1
                    succ[s].append(float(term[LEARNER][i].item()))
                    committed[i] = 0
                    n += 1
                ep_slab = env._slab_top.clone()

        label = "BIT FLIPPED (lying)" if flip else "TRUE BIT"
        print(f"\n=== {label} ===")
        print(f"{'slab side':>12} {'-> top':>8} {'-> bottom':>10} {'neither':>9} "
              f"{'success':>9}")
        for s, name in ((1, "top"), (0, "bottom")):
            tot = max(1, sum(tab[(s, c)] for c in (0, 1, 2)))
            sc = float(np.mean(succ[s])) if succ[s] else float("nan")
            print(f"{name:>12} {tab[(s, 1)] / tot:8.2f} {tab[(s, 2)] / tot:10.2f} "
                  f"{tab[(s, 0)] / tot:9.2f} {sc:9.2f}")
        print("  (correct route avoids the slab: slab top -> bottom corridor)")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
