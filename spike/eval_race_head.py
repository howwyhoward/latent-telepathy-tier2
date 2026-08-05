"""WP1 — deterministic evaluation of frozen race-v8 heads under message corruption.

Training-time route-optimality is measured on SAMPLED decisions with entropy
regularization still on the books; a referee will ask what the head actually
knows. This eval freezes everything (executor mean actions, head argmax) and
re-measures under five wire conditions:

  intact       message as delivered by the bus.
  zero_content anchor dims kept, broadcast content zeroed. Tests that the
               decision comes from the CONTENT, not from the geometry anchor.
  zero_all     whole wire zeroed (the v8 `none` floor, now at eval time).
  shuffle      each env receives another env's message (random permutation at
               the decide step). Marginal message statistics preserved,
               correlation with own slab destroyed.
  noise        content dims replaced with unit Gaussian noise, anchor kept.

Pre-registered expectations (corrected from the work spec: THIS executor balks
at slabs rather than crossing them, so hazard-steps stay ~0 in every mode and
degradation must appear in route-optimality and success):

  intact                  route_opt within ~0.03 of the training final.
  zero_content/zero_all   route_opt ~0.5. Argmax of a constant logit vector is
                          a single fixed corridor, so route_opt = P(that
                          corridor is safe) ~ 0.5 and top-fraction is 0 or 1.
  shuffle/noise           route_opt ~0.5, top-fraction strictly between 0 and
                          1 (decisions vary but carry no slab information).
  all modes               hazard << a crossing (~20 steps): a corrupted head
                          sends the executor to the slab where it may clip
                          1-2 hazard steps while balking, never ~20 crossing.
                          Obedience ~1 (the executor is intact).

Estimator amendment (diagnosed on the first run, hypothesis unchanged): a
completion-stream counter is length-biased — correct decisions finish in ~206
steps, wrong ones in ~380+, so fast (correct) episodes are over-collected and
corrupted modes read 0.55-0.64 instead of ~0.5. Each env therefore contributes
exactly its first K=episodes/num_envs episodes, equal weight, no throughput
bias.

    python spike/eval_race_head.py --condition z_t \
        --head runs/race_v8/z_t.pt --out runs/diag/eval_race_head_z_t_s1.json
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default="z_t",
                    choices=["z_t", "oracle"])
parser.add_argument("--head", type=str, default="runs/race_v8/z_t.pt")
parser.add_argument("--executor", type=str, default="runs/route_obey_v6/cont.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0)
parser.add_argument("--episodes", type=int, default=256, help="per mode")
parser.add_argument("--modes", type=str,
                    default="intact,zero_content,zero_all,shuffle,noise")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import time
import traceback


def _die_loudly(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die_loudly

import numpy as np
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
ANCHOR_DIMS = 2
WIRE = LATENT_DIM + ANCHOR_DIMS
DECIDE_STEP = 2  # same reset-frame-staleness guard as the trainer


def corrupt(m: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "intact":
        return m
    if mode == "zero_content":
        out = m.clone()
        out[:, ANCHOR_DIMS:] = 0.0
        return out
    if mode == "zero_all":
        return torch.zeros_like(m)
    if mode == "shuffle":
        perm = torch.randperm(m.shape[0], device=m.device)
        return m[perm]
    if mode == "noise":
        out = m.clone()
        out[:, ANCHOR_DIMS:] = torch.randn_like(out[:, ANCHOR_DIMS:])
        return out
    raise ValueError(mode)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    # identical env regime to the v8 trainer
    cfg.route_instruction = True
    cfg.route_shaping = True
    cfg.route_abort_wrong = False
    cfg.rew_wrong_corridor = 0.0
    env = ChokepointEnv(cfg)
    device = env.device
    E = args.num_envs

    ck = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ck["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    executor = AttentionReceiver(
        encoder, broadcast_dim=0, latent_dim=LATENT_DIM, route_dim=ROUTE_DIM
    ).to(device)
    executor.load_state_dict(torch.load(args.executor, map_location=device)["policy"])
    executor.eval()

    head = RouteHead(WIRE).to(device)
    head.load_state_dict(torch.load(args.head, map_location=device)["head"])
    head.eval()

    if args.condition == "oracle":
        bus = OracleBroadcast(
            comm_radius=args.comm_radius, broadcast_dim=LATENT_DIM, anchored=True
        )
    else:
        bus = LatentBroadcast(
            encoder, comm_radius=args.comm_radius,
            broadcast_dim=LATENT_DIM, anchored=True,
        )

    def msg_vec() -> torch.Tensor:
        messages, mask = bus.deliver(env)[LEARNER]
        return torch.nan_to_num(messages[:, 0, :] * mask[:, 0:1].float())

    empty_msg = torch.zeros(E, 0, 1, device=device)
    empty_mask = torch.zeros(E, 0, device=device)
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)

    quota = max(1, args.episodes // E)  # per-env cap kills length bias
    results = {}
    for mode in args.modes.split(","):
        env.reset()
        obs_dict = None
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step({LEARNER: zero_scout, BEACON: zero_scout})

        ep_t = torch.zeros(E, dtype=torch.long, device=device)
        acc_haz = torch.zeros(E, device=device)
        choice = torch.zeros(E, dtype=torch.long, device=device)
        first_corr = torch.zeros(E, dtype=torch.long, device=device)
        dec_safe_top = torch.zeros(E, dtype=torch.bool, device=device)
        dec_bad = torch.zeros(E, dtype=torch.bool, device=device)
        n_done = torch.zeros(E, dtype=torch.long, device=device)

        route_opt, succ, haz, obey, top_frac, steps = [], [], [], [], [], []
        n_bad = 0
        t0 = time.time()
        while int((n_done >= quota).sum()) < E:
            m = corrupt(msg_vec(), mode)
            with torch.no_grad():
                logits, _ = head(m)
            greedy = logits.argmax(dim=-1)
            deciding = ep_t <= DECIDE_STEP
            choice = torch.where(deciding, greedy, choice)
            commit = ep_t == DECIDE_STEP
            if commit.any():
                dec_safe_top[commit] = ~env._slab_top[commit]
                dec_bad[commit] = False

            route_top = choice == 0
            env._route_top[:] = route_top
            route = torch.zeros(E, ROUTE_DIM, device=device)
            route[route_top, 0] = 1.0
            route[~route_top, 1] = 1.0

            rgb = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                action = executor.actor(
                    executor.features(rgb, empty_msg, empty_mask, route)
                )
            action = torch.nan_to_num(action)
            obs_dict, rew, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            dec_bad |= ~torch.isfinite(rew[LEARNER])
            terminated = term[LEARNER]
            done = terminated | tout[LEARNER]
            alive = ~done
            acc_haz += env._in_hazard(LEARNER).float() * alive.float()
            in_top = env.in_corridor(LEARNER, top=True) & alive
            in_bot = env.in_corridor(LEARNER, top=False) & alive
            fresh = (first_corr == 0) & (in_top | in_bot)
            first_corr = torch.where(
                fresh, torch.where(in_top, 1, 2).long(), first_corr
            )
            ep_t += 1

            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    if int(n_done[i]) >= quota:
                        continue
                    n_done[i] += 1
                    if bool(dec_bad[i]):
                        n_bad += 1
                        continue
                    top_chosen = int(choice[i].item()) == 0
                    route_opt.append(float(top_chosen == bool(dec_safe_top[i])))
                    succ.append(float(terminated[i].item()))
                    haz.append(float(acc_haz[i].item()))
                    want = 1 if top_chosen else 2
                    obey.append(float(int(first_corr[i].item()) == want))
                    top_frac.append(float(top_chosen))
                    steps.append(int(ep_t[i].item()))
                ep_t[idx] = 0
                acc_haz[idx] = 0.0
                first_corr[idx] = 0
                dec_bad[idx] = False

        row = dict(
            episodes=len(succ),
            route_opt=float(np.mean(route_opt)),
            success=float(np.mean(succ)),
            hazard=float(np.mean(haz)),
            obey=float(np.mean(obey)),
            top_frac=float(np.mean(top_frac)),
            mean_steps=float(np.mean(steps)),
            bad=n_bad,
            secs=round(time.time() - t0, 1),
        )
        results[mode] = row
        print(f"[eval/{args.condition}/{mode:12s}] "
              f"route_opt {row['route_opt']:.3f}  succ {row['success']:.3f}  "
              f"hazard {row['hazard']:5.2f}  obey {row['obey']:.3f}  "
              f"top_frac {row['top_frac']:.2f}  steps {row['mean_steps']:.0f}  "
              f"bad {n_bad}  ({row['secs']}s)")
        sys.stdout.flush()

    print("\nPRE-REGISTRATION CHECK")
    ok = True
    if "intact" in results:
        good = results["intact"]["route_opt"] >= 0.95
        ok &= good
        print(f"  intact route_opt >= 0.95: "
              f"{results['intact']['route_opt']:.3f}  {'PASS' if good else 'FAIL'}")
    for mode in ("zero_content", "zero_all", "shuffle", "noise"):
        if mode in results:
            good = abs(results[mode]["route_opt"] - 0.5) <= 0.10
            ok &= good
            print(f"  {mode} route_opt ~ 0.5 (+/-0.10): "
                  f"{results[mode]['route_opt']:.3f}  {'PASS' if good else 'FAIL'}")
    # A crossing costs ~20 hazard steps; a balk that clips the slab edge costs
    # 1-2. The bound must separate those regimes, not demand exact zero.
    worst_haz = max(r["hazard"] for r in results.values())
    good = worst_haz <= 3.0
    ok &= good
    print(f"  hazard << crossing (~20) in ALL modes (executor balks): "
          f"worst {worst_haz:.2f}  {'PASS' if good else 'FAIL'}")
    print(f"WP1 GATE: {'PASS' if ok else 'FAIL — diagnose the deviating mode'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "args": {
                k: v for k, v in vars(args).items() if isinstance(
                    v, (int, float, str, bool, type(None)))
            }}, f, indent=2)
        print(f"wrote -> {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
