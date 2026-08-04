"""Watch a trained navigator drive the chokepoint, live over WebRTC.

Unlike quick_look.py (which just parks the scene so you can inspect geometry),
this replays an actual policy and narrates what it is doing: which corridor it
committed to, whether it backed out, whether it touched the slab, and how the
episode ended. Works with any message condition, so the same script shows the
blind trunk and, later, an informed race policy.

On wulab1 (inside tmux), with nothing else holding the stream ports:

    source setup/env.sh
    LIVESTREAM=2 python spike/watch_policy.py                      # blind trunk
    LIVESTREAM=2 python spike/watch_policy.py --slab top           # pin the slab
    LIVESTREAM=2 python spike/watch_policy.py \
        --policy runs/race_v6/oracle_g999.pt --condition oracle    # informed

Then connect the Mac WebRTC client to 10.45.7.145 (TCP 49100 / UDP 47998).
Only ONE Isaac streaming process may run at a time — two both bind 49100 and
the client renders black.

--slab {top,bottom} pins the hazard so you can watch the same decision
repeatedly; --slab flip (default) keeps the training coin flip.
Add --headless to run without streaming (narration only, e.g. over ssh).
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str,
                    default="runs/nav_pretrain/nav_s1_mouth.pt",
                    help="checkpoint to drive the navigator with")
parser.add_argument("--condition", type=str, default="none",
                    choices=["none", "position", "z_t", "raw", "oracle"],
                    help="must match the condition the checkpoint was trained in")
parser.add_argument("--slab", type=str, default="flip",
                    choices=["flip", "top", "bottom"])
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0)
parser.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the policy mean")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.message_bus import (  # noqa: E402
    LatentBroadcast,
    MessageBus,
    OracleBroadcast,
    RawObsBroadcast,
)
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"


def make_bus(condition, encoder):
    if condition == "none":
        return MessageBus(comm_radius=args.comm_radius, broadcast_dim=0)
    if condition == "position":
        return MessageBus(comm_radius=args.comm_radius,
                          broadcast_dim=LATENT_DIM, anchored=True)
    if condition == "z_t":
        return LatentBroadcast(encoder, comm_radius=args.comm_radius,
                               broadcast_dim=LATENT_DIM, anchored=True)
    if condition == "raw":
        return RawObsBroadcast(comm_radius=args.comm_radius, anchored=True)
    return OracleBroadcast(comm_radius=args.comm_radius,
                           broadcast_dim=LATENT_DIM, anchored=True)


def where(x: float, y: float) -> str:
    """Human-readable zone, so the console narrates what the viewport shows."""
    lane = "top" if y > 0.5 else ("bottom" if y < -0.5 else "mid")
    if x < -3.0:
        return "west chamber"
    if x > 3.0:
        return "east chamber"
    return f"{lane} corridor" if lane != "mid" else "between corridors"


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = 1
    cfg.success_agents = [LEARNER]
    if args.slab != "flip":
        cfg.force_slab_top = args.slab == "top"
    env = ChokepointEnv(cfg)
    device = env.device
    obs, _ = env.reset()

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    bus = make_bus(args.condition, encoder)
    policy = AttentionReceiver(
        encoder, broadcast_dim=bus.wire_dim, latent_dim=LATENT_DIM
    ).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device)["policy"])
    policy.eval()

    zero_scout = torch.zeros(1, N_ACTIONS, device=device)
    goal = env._goal_pos[LEARNER][0]

    print(f"[watch] policy   : {args.policy}")
    print(f"[watch] condition: {args.condition} (wire_dim {bus.wire_dim})")
    print(f"[watch] slab     : {args.slab}")
    print("[watch] scene up — connect the WebRTC client now\n")

    episode, step, touched, committed = 1, 0, False, None
    slab_side = "TOP" if env._slab_top[0] else "BOTTOM"
    print(f"[ep {episode}] slab in {slab_side} corridor — scout "
          f"{'SEES it' if env._slab_top[0] else 'sees a CLEAN top corridor'}")

    while simulation_app.is_running():
        rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
        msgs, mask = bus.deliver(env)[LEARNER]
        with torch.no_grad():
            if args.stochastic:
                action, _, _, _ = policy.get_action_and_value(rgb, msgs, mask.float())
            else:
                action = policy.actor(policy.features(rgb, msgs, mask.float()))
        obs, _, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        step += 1

        pos = env._local_pos(LEARNER)[0]
        x, y = float(pos[0]), float(pos[1])
        if env._in_hazard(LEARNER)[0]:
            touched = True
        # first corridor entry = the decision the message is supposed to inform
        if committed is None and x > -3.0 and abs(y) > 0.5:
            committed = "top" if y > 0 else "bottom"
            safe = (committed == "top") != bool(env._slab_top[0])
            print(f"[ep {episode}] step {step:3d}: committed to the {committed} "
                  f"corridor — {'CLEAN' if safe else 'SLABBED'}")
        if step % 20 == 0:
            d = float(torch.norm(pos - goal))
            print(f"[ep {episode}] step {step:3d}: {where(x, y):<17} "
                  f"({x:+.2f}, {y:+.2f})  goal {d:4.1f} m"
                  f"{'  IN HAZARD' if env._in_hazard(LEARNER)[0] else ''}")

        if (term[LEARNER] | tout[LEARNER])[0]:
            outcome = "REACHED GOAL" if term[LEARNER][0] else "timed out"
            print(f"[ep {episode}] {outcome} after {step} steps; "
                  f"slab was {slab_side}, took {committed} corridor, "
                  f"{'crossed the slab' if touched else 'never touched the slab'}\n")
            episode, step, touched, committed = episode + 1, 0, False, None
            slab_side = "TOP" if env._slab_top[0] else "BOTTOM"
            print(f"[ep {episode}] slab in {slab_side} corridor")


if __name__ == "__main__":
    main()
