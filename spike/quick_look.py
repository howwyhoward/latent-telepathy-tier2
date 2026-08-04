"""Interactive look at the chokepoint scene via WebRTC livestream.

Current arena (race v5): the Tier 1 inter-corridor RUNG IS SEALED
(geometry.remove_rung) — the wall between the corridors is now solid for the
full span, so corridor choice is a real commitment. Watch the red slab hop
between the top/bottom corridor on each reset (printed to this console).

On wulab1 (inside tmux):

    source setup/env.sh
    LIVESTREAM=2 python spike/quick_look.py             # robots idle
    LIVESTREAM=2 python spike/quick_look.py --random    # random cmd_vel
    LIVESTREAM=2 python spike/quick_look.py \
        --policy runs/race_v3/z_t_s1.pt                 # replay a trained navigator

Wait for the app-loaded message, then connect the Mac WebRTC client to
10.45.7.145 (TCP 49100 signaling + UDP 47998 media, hardcoded in the client).

Useful in-viewport checks: select Navigator/cam or Scout/cam in the stage
tree and switch the viewport to that camera — the onboard view should match
the offscreen frames the gate measures. Episodes time out after 60 s.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true", help="drive with random cmd_vel")
parser.add_argument("--policy", type=str, default=None,
                    help="race checkpoint (.pt): drive the navigator with it (z_t bus)")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
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
obs, _ = env.reset()

policy = bus = None
if args.policy:
    from chokepoint.constants import LATENT_DIM  # noqa: E402
    from chokepoint.jepa import PixelEncoder  # noqa: E402
    from chokepoint.message_bus import LatentBroadcast  # noqa: E402
    from chokepoint.receiver import AttentionReceiver  # noqa: E402

    ckpt = torch.load(args.jepa_ckpt, map_location=env.device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(env.device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    bus = LatentBroadcast(encoder, comm_radius=12.0,
                          broadcast_dim=LATENT_DIM, anchored=True)
    blob = torch.load(args.policy, map_location=env.device)
    policy = AttentionReceiver(
        encoder, broadcast_dim=bus.wire_dim, latent_dim=LATENT_DIM
    ).to(env.device)
    policy.load_state_dict(blob["policy"])
    policy.eval()
    print(f"[look] driving navigator with {args.policy} "
          f"(condition {blob.get('condition', '?')})")

print("[look] scene up — connect the WebRTC client now")


def slab_side() -> str:
    return "TOP" if env._slab_top[0] else "BOTTOM"


print(f"[look] episode 1: hazard slab in {slab_side()} corridor")
episode = 1

while simulation_app.is_running():
    zeros = torch.zeros(env.num_envs, 3, device=env.device)
    if policy is not None:
        rgb = obs["navigator"].permute(0, 3, 1, 2).contiguous()
        msgs, mask = bus.deliver(env)["navigator"]
        with torch.no_grad():
            action = policy.actor(policy.features(rgb, msgs, mask.float()))
        actions = {"navigator": action.clamp(-1, 1), "scout": zeros}
    elif args.random:
        actions = {
            a: torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
            for a in env.cfg.possible_agents
        }
    else:
        actions = {a: zeros for a in env.cfg.possible_agents}
    obs, _, term, tout, _ = env.step(actions)
    if (term["navigator"] | tout["navigator"]).any():
        episode += 1
        outcome = "SUCCESS" if term["navigator"][0] else "timeout"
        print(f"[look] episode {episode}: previous ended in {outcome} — "
              f"hazard slab now in {slab_side()} corridor")
