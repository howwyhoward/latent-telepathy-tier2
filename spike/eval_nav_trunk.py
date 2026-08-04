"""Region-wise competence check for a stage-1 navigation trunk.

Stage 1 (random-spawn pretraining) reports one aggregate success number, but
the race only needs competence on the actual route: west chamber (the choice
point), inside each corridor, and the east chamber. This teleports the
navigator to controlled poses after reset and measures per-region success
with the deterministic policy.

    python spike/eval_nav_trunk.py --policy runs/nav_pretrain/nav_s1.pt
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="runs/nav_pretrain/nav_s1.pt")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes_per_region", type=int, default=32)
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.geometry import ROBOT_SIZE  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"

# (name, x, y, yaw) in env-local meters; yaw 0 = east (toward the goal side).
# Corridor y-centers: top +1.25, bottom -1.25 (rows 7/8 and 12/13 of the
# 20-cell grid at 0.5 m/cell). Canonical start is the Tier 1 pose.
REGIONS = [
    ("canonical_start", -3.75, -0.25, 0.0),
    ("west_chamber_hi", -4.0, 1.5, 0.0),
    ("top_corridor_mouth", -2.5, 1.25, 0.0),
    ("top_corridor_mid", 0.0, 1.25, 0.0),
    ("bottom_corridor_mouth", -2.5, -1.25, 0.0),
    ("bottom_corridor_mid", 0.0, -1.25, 0.0),
    ("east_chamber", 3.75, 1.5, -math.pi / 2),
]


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device
    env.reset()

    ckpt = torch.load("checkpoints/jepa_pixels.pt", map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    blob = torch.load(args.policy, map_location=device)
    policy = AttentionReceiver(encoder, broadcast_dim=0, latent_dim=LATENT_DIM).to(device)
    policy.load_state_dict(blob["policy"])
    policy.eval()

    zero_scout = torch.zeros(args.num_envs, N_ACTIONS, device=device)
    empty_msg = torch.zeros(args.num_envs, 0, 1, device=device)
    empty_mask = torch.zeros(args.num_envs, 0, device=device)

    print(f"policy: {args.policy}  (condition {blob.get('condition', '?')})")
    print(f"{'region':>22}  {'success':>7}  {'mean_len':>8}  {'hazard':>6}")

    for name, x, y, yaw in REGIONS:
        done_count, succ, lens, haz = 0, [], [], []
        env.reset()
        # teleport the navigator to the region pose in every env
        pose = torch.zeros(args.num_envs, 7, device=device)
        pose[:, 0] = env.scene.env_origins[:, 0] + x
        pose[:, 1] = env.scene.env_origins[:, 1] + y
        pose[:, 2] = ROBOT_SIZE[2] / 2
        pose[:, 3] = math.cos(yaw / 2)
        pose[:, 6] = math.sin(yaw / 2)
        env.scene[LEARNER].write_root_pose_to_sim(pose)
        env.scene[LEARNER].write_root_velocity_to_sim(
            torch.zeros(args.num_envs, 6, device=device)
        )
        acc_h = torch.zeros(args.num_envs, device=device)
        acc_l = torch.zeros(args.num_envs, device=device)
        obs, *_ = env.step({LEARNER: torch.zeros(args.num_envs, 3, device=device),
                            BEACON: zero_scout})
        while done_count < args.episodes_per_region:
            rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                action = policy.actor(policy.features(rgb, empty_msg, empty_mask))
            obs, _, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            acc_h += env._in_hazard(LEARNER).float()
            acc_l += 1
            done = term[LEARNER] | tout[LEARNER]
            if done.any():
                for i in done.nonzero(as_tuple=True)[0].tolist():
                    succ.append(float(term[LEARNER][i].item()))
                    lens.append(acc_l[i].item())
                    haz.append(acc_h[i].item())
                    acc_h[i] = 0.0
                    acc_l[i] = 0.0
                    done_count += 1
                # note: auto-reset puts done envs back at canonical start;
                # those episodes still count toward canonical competence but
                # we stop scoring the region once quota is met, so keep the
                # region pure by re-teleporting the done envs
                idx = done.nonzero(as_tuple=True)[0]
                p = pose[idx].clone()
                p[:, 0] = env.scene.env_origins[idx, 0] + x
                p[:, 1] = env.scene.env_origins[idx, 1] + y
                env.scene[LEARNER].write_root_pose_to_sim(p, env_ids=idx)
        import numpy as np

        print(f"{name:>22}  {np.mean(succ):7.2f}  {np.mean(lens):8.1f}  "
              f"{np.mean(haz):6.1f}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
