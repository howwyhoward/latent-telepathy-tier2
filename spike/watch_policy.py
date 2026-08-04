"""Watch the full race-v8 system drive the chokepoint, live over WebRTC.

This replays the CURRENT BEST composition — the thing the RoboMasters will
run — and narrates every stage of it:

  scout camera -> frozen JEPA encoder -> z_t on the wire
      -> route head (trained by task reward alone, race v8)
      -> 1-bit route command
      -> frozen stage-1.5 executor (route obedience, gate-passing cont.pt)

Per episode you will see: where the slab is, what the head decoded from the
message (with its confidence), whether that was the safe corridor, the moment
the navigator commits, and how the episode ends. Expect: a beeline west out
of the east chamber, a committed turn into the commanded corridor at the
chamber mouth, a clean traversal, and a goal stop — with the corridor
flipping whenever the slab does. That corridor flip IS the recruitment
result, watched live.

On wulab1 (inside tmux), with nothing else holding the stream ports:

    source setup/env.sh
    LIVESTREAM=2 python spike/watch_policy.py                    # z_t, the thesis
    LIVESTREAM=2 python spike/watch_policy.py --slab top         # pin the slab
    LIVESTREAM=2 python spike/watch_policy.py --condition oracle \
        --head runs/race_v8/oracle.pt                            # ceiling control
    LIVESTREAM=2 python spike/watch_policy.py --route top        # bypass the head

Then connect the Mac WebRTC client to 10.45.7.145 (TCP 49100 / UDP 47998).
Only ONE Isaac streaming process may run at a time — two both bind 49100 and
the client renders black. Add --headless for narration-only over ssh.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--executor", type=str, default="runs/route_obey_v6/cont.pt",
                    help="frozen stage-1.5 route-obedience policy")
parser.add_argument("--head", type=str, default="runs/race_v8/z_t_s3.pt",
                    help="race-v8 route head (message -> corridor)")
parser.add_argument("--condition", type=str, default="z_t",
                    choices=["none", "z_t", "oracle"],
                    help="must match the condition the head was trained in")
parser.add_argument("--route", type=str, default="head",
                    choices=["head", "top", "bottom"],
                    help="who commands the corridor: the head, or you")
parser.add_argument("--slab", type=str, default="flip",
                    choices=["flip", "top", "bottom"])
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0)
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
from chokepoint.message_bus import LatentBroadcast, OracleBroadcast  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402
from chokepoint.route_head import RouteHead  # noqa: E402

LEARNER, BEACON = "navigator", "scout"
ROUTE_DIM = 2
WIRE = LATENT_DIM + 2
DECIDE_STEP = 2  # reset-time camera frames are stale; decode on a live one


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

    executor = AttentionReceiver(
        encoder, broadcast_dim=0, latent_dim=LATENT_DIM, route_dim=ROUTE_DIM
    ).to(device)
    executor.load_state_dict(torch.load(args.executor, map_location=device)["policy"])
    executor.eval()

    head = RouteHead(WIRE).to(device)
    if args.route == "head":
        head.load_state_dict(torch.load(args.head, map_location=device)["head"])
    head.eval()

    if args.condition == "oracle":
        bus = OracleBroadcast(comm_radius=args.comm_radius,
                              broadcast_dim=LATENT_DIM, anchored=True)
    elif args.condition == "z_t":
        bus = LatentBroadcast(encoder, comm_radius=args.comm_radius,
                              broadcast_dim=LATENT_DIM, anchored=True)
    else:
        bus = None

    def msg_vec() -> torch.Tensor:
        if bus is None:
            return torch.zeros(1, WIRE, device=device)
        messages, mask = bus.deliver(env)[LEARNER]
        return torch.nan_to_num(messages[:, 0, :] * mask[:, 0:1].float())

    empty_msg = torch.zeros(1, 0, 1, device=device)
    empty_mask = torch.zeros(1, 0, device=device)
    zero_scout = torch.zeros(1, N_ACTIONS, device=device)
    goal = env._goal_pos[LEARNER][0]

    print(f"[watch] executor : {args.executor}")
    print(f"[watch] head     : {args.head if args.route == 'head' else '(manual)'}"
          f"  condition {args.condition}")
    print(f"[watch] slab     : {args.slab}   route: {args.route}")
    print("[watch] scene up — connect the WebRTC client now\n")

    episode, step, touched, committed = 1, 0, False, None
    route_top = None  # decided at DECIDE_STEP
    route = torch.zeros(1, ROUTE_DIM, device=device)  # route-blind until decided
    slab_side = "TOP" if env._slab_top[0] else "BOTTOM"
    print(f"[ep {episode}] slab in {slab_side} corridor — safe route is "
          f"{'BOTTOM' if env._slab_top[0] else 'TOP'}")

    while simulation_app.is_running():
        if step == DECIDE_STEP:
            if args.route == "head":
                with torch.no_grad():
                    logits, _ = head(msg_vec())
                    p = torch.softmax(logits, dim=-1)[0]
                route_top = bool(p[0] > p[1])
                conf = float(p.max())
                verdict = ("CORRECT" if route_top != bool(env._slab_top[0])
                           else "WRONG — heading for the slab")
                print(f"[ep {episode}] step {step:3d}: head decodes the "
                      f"{args.condition} message -> commands "
                      f"{'TOP' if route_top else 'BOTTOM'} (p={conf:.2f})  "
                      f"[{verdict}]")
            else:
                route_top = args.route == "top"
                print(f"[ep {episode}] step {step:3d}: manual route command: "
                      f"{'TOP' if route_top else 'BOTTOM'}")
            route = torch.zeros(1, ROUTE_DIM, device=device)
            route[0, 0 if route_top else 1] = 1.0

        rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            action = executor.actor(
                executor.features(rgb, empty_msg, empty_mask, route)
            )
        obs, _, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        step += 1

        pos = env._local_pos(LEARNER)[0]
        x, y = float(pos[0]), float(pos[1])
        if env._in_hazard(LEARNER)[0]:
            touched = True
        # first corridor entry = the decision the message informed
        if committed is None and x > -3.0 and abs(y) > 0.5:
            committed = "top" if y > 0 else "bottom"
            safe = (committed == "top") != bool(env._slab_top[0])
            obeyed = route_top is not None and (committed == "top") == route_top
            print(f"[ep {episode}] step {step:3d}: entered the {committed} "
                  f"corridor — {'CLEAN' if safe else 'SLABBED'}"
                  f"{', as commanded' if obeyed else ', DISOBEYING the command'}")
        if step % 20 == 0:
            d = float(torch.norm(pos - goal))
            print(f"[ep {episode}] step {step:3d}: {where(x, y):<17} "
                  f"({x:+.2f}, {y:+.2f})  goal {d:4.1f} m"
                  f"{'  IN HAZARD' if env._in_hazard(LEARNER)[0] else ''}")

        if (term[LEARNER] | tout[LEARNER])[0]:
            outcome = "REACHED GOAL" if term[LEARNER][0] else "timed out"
            print(f"[ep {episode}] {outcome} after {step} steps; slab was "
                  f"{slab_side}, commanded "
                  f"{'TOP' if route_top else 'BOTTOM' if route_top is not None else '—'}, "
                  f"took {committed} corridor, "
                  f"{'crossed the slab' if touched else 'never touched the slab'}\n")
            episode, step, touched, committed = episode + 1, 0, False, None
            route_top = None
            route = torch.zeros(1, ROUTE_DIM, device=device)
            slab_side = "TOP" if env._slab_top[0] else "BOTTOM"
            print(f"[ep {episode}] slab in {slab_side} corridor — safe route is "
                  f"{'BOTTOM' if env._slab_top[0] else 'TOP'}")


if __name__ == "__main__":
    main()
