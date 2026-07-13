"""Interactive look at the chokepoint scene via WebRTC livestream.

On wulab1 (inside tmux):

    source setup/env.sh
    LIVESTREAM=2 python spike/quick_look.py            # robots idle
    LIVESTREAM=2 python spike/quick_look.py --random   # random cmd_vel

Wait for the app-loaded message, then connect the Mac WebRTC client to
10.45.7.145 (TCP 49100 signaling + UDP 47998 media, hardcoded in the client).

Useful in-viewport checks: select Navigator/cam or Scout/cam in the stage
tree and switch the viewport to that camera — the onboard view should match
the offscreen frames the gate measures. Episodes reset every 30 s; watch the
red slab hop between corridors.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true", help="drive with random cmd_vel")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402

cfg = ChokepointEnvCfg()
cfg.scene.num_envs = 1  # one arena — this is for looking at, not training
env = ChokepointEnv(cfg)
env.reset()
print("[look] scene up — connect the WebRTC client now")

while simulation_app.is_running():
    if args.random:
        actions = {
            a: torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
            for a in env.cfg.possible_agents
        }
    else:
        actions = {
            a: torch.zeros(env.num_envs, 3, device=env.device)
            for a in env.cfg.possible_agents
        }
    env.step(actions)
