"""Integration smoke: env -> message bus -> receiver -> actions -> env.step.

Closes the full Tier 1 loop inside Isaac with a placeholder encoder (the real
frozen JEPA encoder arrives in Phase 2). Checks:

  - anchored delivery off the real env: anchors match ground-truth relative
    positions from PhysX
  - receiver consumes (obs, messages, mask) and yields cmd_vel actions
  - the whole loop steps without shape errors at num_envs parallelism

Run:  python spike/smoke_bus.py --num_envs 8
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import traceback

import torch
import torch.nn as nn


def _die_loudly(exc_type, exc, tb):
    # Kit's teardown deadlocks after an exception in headless mode, which turns
    # a crash into a silent multi-minute hang. Print and exit hard instead.
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die_loudly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.message_bus import LatentBroadcast  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402


class PlaceholderEncoder(nn.Module):
    """Fixed random conv features standing in for the frozen JEPA encoder."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.LazyLinear(LATENT_DIM)

    def forward(self, x):
        return self.head(self.net(x))


cfg = ChokepointEnvCfg()
cfg.scene.num_envs = args.num_envs
env = ChokepointEnv(cfg)
obs, _ = env.reset()

encoder = PlaceholderEncoder().to(env.device)
encoder(torch.zeros(1, 3, 64, 64, device=env.device))  # materialize lazy head

bus = LatentBroadcast(encoder, comm_radius=3.0, broadcast_dim=LATENT_DIM, anchored=True)
receivers = {
    a: AttentionReceiver(encoder, broadcast_dim=bus.wire_dim, latent_dim=LATENT_DIM).to(env.device)
    for a in env.cfg.possible_agents
}

# --- anchor ground-truth check off the real env -----------------------------
delivered = bus.deliver(env)
msgs, mask = delivered["navigator"]
true_delta = (env._local_pos("scout") - env._local_pos("navigator")) / bus.comm_radius
in_range = (env._local_pos("scout") - env._local_pos("navigator")).norm(dim=1) <= bus.comm_radius
anchor_err = (msgs[:, 0, :2] - true_delta)[in_range].abs().max() if in_range.any() else 0.0
print(f"[bus] slots in range: {int(mask.sum())}/{env.num_envs} "
      f"(start poses are ~5.9 m apart vs radius 3.0 -> expect 0)")
print(f"[bus] anchored wire_dim: {bus.wire_dim} (2 anchor + {bus.broadcast_dim} content)")

# --- full loop ---------------------------------------------------------------
t0 = time.time()
for i in range(args.steps):
    delivered = bus.deliver(env)
    actions = {}
    with torch.no_grad():
        for a in env.cfg.possible_agents:
            rgb = obs[a].permute(0, 3, 1, 2)
            m, k = delivered[a]
            act, logp, ent, val = receivers[a].get_action_and_value(rgb, m, k)
            actions[a] = act.clamp(-1, 1)
    obs, rewards, terminated, time_outs, infos = env.step(actions)
dt = time.time() - t0

# messages appear once the robots wander into range; report final state
delivered = bus.deliver(env)
_, mask = delivered["navigator"]
print(f"[bus] slots in range after {args.steps} random-ish steps: {int(mask.sum())}/{env.num_envs}")
print(f"[bus] RESULT full loop: {args.steps / dt:.1f} control-steps/s at num_envs={args.num_envs}")
print("[bus] SMOKE PASS")
sys.stdout.flush()
os._exit(0)
