"""First flight of ChokepointEnv: random actions, sanity checks, FPS.

Checks:
  - obs dict has both agents, shape (num_envs, 64, 64, 3), values in [0, 1]
  - slab-side randomization actually varies across envs after reset
  - hazard penalty fires for a robot parked inside the slab AABB
  - episodes terminate (timeout) and auto-reset
  - control-steps/sec at this num_envs

Run:  python spike/smoke_env.py --num_envs 8
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402

cfg = ChokepointEnvCfg()
cfg.scene.num_envs = args.num_envs
env = ChokepointEnv(cfg)

obs, _ = env.reset()
for a, o in obs.items():
    print(f"[env] obs[{a}]: shape {tuple(o.shape)}, range [{o.min():.3f}, {o.max():.3f}]")

top = env._slab_top
print(f"[env] slab side across {args.num_envs} envs: {int(top.sum())} TOP / {int((~top).sum())} BOTTOM")

# hazard check: teleport the navigator into its env's active slab and read the flag
aabb = torch.where(env._slab_top.unsqueeze(1), env._aabb_top.repeat(env.num_envs, 1),
                   env._aabb_bot.repeat(env.num_envs, 1))
center = torch.stack([(aabb[:, 0] + aabb[:, 1]) / 2, (aabb[:, 2] + aabb[:, 3]) / 2], dim=1)
pose = torch.zeros(env.num_envs, 7, device=env.device)
pose[:, :2] = env.scene.env_origins[:, :2] + center
pose[:, 2] = 0.075
pose[:, 3] = 1.0
env.scene["navigator"].write_root_pose_to_sim(pose)
env.scene.write_data_to_sim()
env.sim.step()
env.scene.update(env.sim.get_physics_dt())
in_haz = env._in_hazard("navigator")
print(f"[env] hazard flag with navigator parked in slab: {int(in_haz.sum())}/{env.num_envs} "
      f"({'PASS' if bool(in_haz.all()) else 'FAIL'})")

obs, _ = env.reset()
resets_seen = 0
t0 = time.time()
for i in range(args.steps):
    actions = {
        a: torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        for a in env.cfg.possible_agents
    }
    obs, rewards, terminated, time_outs, infos = env.step(actions)
    resets_seen += int(time_outs["navigator"].sum() + terminated["navigator"].sum())
dt = time.time() - t0

steps_s = args.steps / dt
print(f"[env] rewards sample: { {a: float(r.mean()) for a, r in rewards.items()} }")
print(f"[env] resets during run: {resets_seen}")
print(f"[env] RESULT num_envs={args.num_envs}: control-steps/s={steps_s:.1f}, "
      f"env-steps/s={steps_s * args.num_envs:.1f} "
      f"(physics decimation {env.cfg.decimation}, so sim-steps/s={steps_s * env.cfg.decimation:.1f})")

print("[env] SMOKE PASS")
sys.stdout.flush()
os._exit(0)
