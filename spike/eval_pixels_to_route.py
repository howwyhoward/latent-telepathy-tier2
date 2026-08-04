"""End-to-end composition: scout pixels -> JEPA latent -> decoded slab side ->
route command -> gate-passing obedience policy -> navigation.

Stage 1.5 proved the navigator obeys a 1-bit route command from the canonical
start (cont.pt: canonical obey 0.96/0.985, succ 0.935/0.895). The linear probes
proved the scout's latent linearly encodes slab side. This script composes the
two for the first time: the route bit is DECODED FROM THE SCOUT'S CAMERA, not
read from the simulator, so a success here is the full perception-to-decision-
to-navigation pipeline running on pixels.

Honest framing: the decoder is a supervised logistic probe, so this is the
"Stage 2a" engineering demonstration (is the pipeline sufficient?), not the
emergent-communication claim (does RL recruit the message on its own?), which
remains the race experiment. Deployment-wise this IS the architecture the
RoboMasters would run.

    python spike/eval_pixels_to_route.py --policy runs/route_obey_v6/cont.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="runs/route_obey_v6/cont.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--probe_episodes", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=256)
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
ROUTE_DIM = 2


def scout_latent(encoder, obs_dict):
    rgb = obs_dict[BEACON].permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        return encoder(rgb)


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    cfg.route_instruction = True   # populates _route_top for ground-truth checks
    cfg.route_shaping = False      # eval: no shaping, no abort, no penalty
    cfg.route_abort_wrong = False
    cfg.rew_wrong_corridor = 0.0
    env = ChokepointEnv(cfg)
    device = env.device
    E = args.num_envs

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    policy = AttentionReceiver(
        PixelEncoder(LATENT_DIM), broadcast_dim=0, latent_dim=LATENT_DIM,
        route_dim=ROUTE_DIM,
    ).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device)["policy"])
    policy.eval()

    empty_msg = torch.zeros(E, 1, 2, device=device)
    empty_mask = torch.zeros(E, 1, device=device)
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)

    # ---- phase 1: fit the logistic probe on (scout latent, slab side) -------
    # Labels come from the sim, so this is supervised decoding -- Stage 2a.
    # Cameras render during stepping, so an image read straight off env.reset()
    # is stale relative to the freshly teleported slab (first attempt scored a
    # chance-level probe this way). Warm up a few parked frames after each
    # reset before reading the scout.
    zs, ys = [], []
    seen = 0
    while seen < args.probe_episodes:
        env.reset()
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step({LEARNER: zero_scout, BEACON: zero_scout})
        zs.append(scout_latent(encoder, obs_dict))
        ys.append(env._slab_top.float().clone())
        seen += E
    Z = torch.cat(zs)
    Y = torch.cat(ys)
    n_tr = int(0.7 * len(Z))
    w = torch.zeros(Z.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Z[:n_tr] @ w + b, Y[:n_tr]
        )
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = ((Z @ w + b) > 0).float()
        acc_tr = (pred[:n_tr] == Y[:n_tr]).float().mean().item()
        acc_te = (pred[n_tr:] == Y[n_tr:]).float().mean().item()
    print(f"probe: {len(Z)} samples  train acc {acc_tr:.3f}  held-out acc {acc_te:.3f}")

    # ---- phase 2: run the composed system -----------------------------------
    obs_dict, _ = env.reset()

    def routes_from_pixels():
        """Decode slab side from the scout camera; command the OPPOSITE corridor."""
        z = scout_latent(encoder, obs_dict)
        slab_top_hat = (z @ w + b) > 0
        route_top = ~slab_top_hat  # safe corridor
        r = torch.zeros(E, ROUTE_DIM, device=device)
        r[route_top, 0] = 1.0
        r[~route_top, 1] = 1.0
        return r, route_top

    # settle one rendered frame so the first decode sees live pixels
    obs_dict, _, _, _, _ = env.step({LEARNER: zero_scout, BEACON: zero_scout})

    succ, haz, obey, decode_oks, n = [], [], [], [], 0
    committed = torch.zeros(E, dtype=torch.long, device=device)  # 0 none,1 top,2 bot
    acc_haz = torch.zeros(E, device=device)
    ep_route_top = torch.zeros(E, dtype=torch.bool, device=device)
    ep_decode_ok = torch.zeros(E, device=device)
    while n < args.eval_episodes:
        # Re-decode every step from the live frame; the slab is static within an
        # episode so this is constant per episode, and it keeps just-reset envs
        # honest without tracking render timing. The commanded route is frozen
        # per env at corridor commitment.
        route, route_top_hat = routes_from_pixels()
        undecided = committed == 0
        ep_route_top = torch.where(undecided, route_top_hat, ep_route_top)
        ep_decode_ok = torch.where(
            undecided, (route_top_hat == env._route_top).float(), ep_decode_ok
        )
        rgb = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            action = policy.actor(policy.features(rgb, empty_msg, empty_mask, route))
        obs_dict, _, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        alive = ~(term[LEARNER] | tout[LEARNER])
        in_top = env.in_corridor(LEARNER, top=True) & alive
        in_bot = env.in_corridor(LEARNER, top=False) & alive
        fresh = (committed == 0) & (in_top | in_bot)
        committed = torch.where(fresh, torch.where(in_top, 1, 2).long(), committed)
        acc_haz += env._in_hazard(LEARNER).float() * alive.float()
        done = term[LEARNER] | tout[LEARNER]
        if done.any():
            for i in done.nonzero(as_tuple=True)[0].tolist():
                succ.append(float(term[LEARNER][i].item()))
                haz.append(float(acc_haz[i].item()))
                want = 1 if bool(ep_route_top[i].item()) else 2
                obey.append(float(int(committed[i].item()) == want))
                decode_oks.append(float(ep_decode_ok[i].item()))
                n += 1
            committed[done] = 0
            acc_haz[done] = 0.0

    print(f"\n=== pixels -> route -> navigation ({n} episodes) ===")
    print(f"decode accuracy {np.mean(decode_oks):.3f}")
    print(f"success        {np.mean(succ):.3f}")
    print(f"obeyed decode  {np.mean(obey):.3f}")
    print(f"hazard steps   {np.mean(haz):.2f} per episode")
    print("\nreference: cont.pt with ground-truth route read 0.935/0.895 success; "
          "a blind policy pays ~20 hazard steps or refuses.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
