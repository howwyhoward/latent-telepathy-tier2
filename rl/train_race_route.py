"""Race v8 — recruit the route from the MESSAGE (stage 2 proper).

v7's post-mortem: a corridor preference cannot live in the lateral component
of a Gaussian action stream — the sigma that buys coverage attenuates its own
gradient by 1/sigma^2, and the policy recruited even a perfect oracle bit for
the wrong function (advance-vs-balk, never corridor choice). Stage 1.5 built
the missing abstraction instead: cont.pt executes a 1-bit route command at
0.96/0.985 canonical obedience.

This trainer closes the loop with the smallest possible learner. The executor
is FROZEN (deterministic mean actions). The only trainable module is a route
head: message -> 2 logits, sampled ONCE per episode, credited with the whole
episode's return — a contextual bandit. Exploration of the alternative route
now costs one categorical sample instead of a 14-sigma Gaussian event, which
is the entire point of the decomposition.

Conditions (same wire as race v7, anchored, matched width):
  oracle : ground-truth slab bit on the wire — optimization ceiling. If this
           fails, the bandit machinery is broken, not the representation.
  z_t    : the scout's frozen JEPA latent — THE thesis condition. Reward is
           the only supervision; nobody tells the head what the latent means.
  none   : zero message at matched width — the floor. The head can only learn
           a constant preference, so route-optimality pins at ~0.5 and every
           other episode pays the hazard crossing.

Pre-registered readout, canonical spawns only: route-optimality (head chose
the slab-free corridor) and hazard-steps/episode. Thesis result = z_t ~ oracle
>> none.

    python rl/train_race_route.py --condition z_t \
        --executor runs/route_obey_v6/cont.pt --save runs/race_v8/z_t.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default="z_t",
                    choices=["none", "z_t", "oracle"])
parser.add_argument("--executor", type=str, default="runs/route_obey_v6/cont.pt")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0)
parser.add_argument("--total_episodes", type=int, default=6000)
parser.add_argument("--batch_episodes", type=int, default=256,
                    help="completed episodes per head update")
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--minibatch_size", type=int, default=64)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
# A 4.5k-param bandit head is not a pixel policy: at 3e-4 the oracle run's
# logits had moved ~0.03 after 7 updates (112 Adam steps) — visibly flat.
parser.add_argument("--learning_rate", type=float, default=3e-3)
# Once V(msg) converges, normalized advantages amplify pure noise to +/-1 and
# a constant lr walks the logits around the optimum (z_t oscillated 0.94 ->
# 0.87 mid-run). Linear decay to 0 consolidates the learned decoding.
parser.add_argument("--anneal_lr", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_csv", type=str, default=None)
parser.add_argument("--run_json", type=str, default=None)
parser.add_argument("--save", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import time
import traceback
from collections import deque


def _die_loudly(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die_loudly

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.message_bus import (  # noqa: E402
    LatentBroadcast,
    MessageBus,
    OracleBroadcast,
)
from chokepoint.receiver import AttentionReceiver  # noqa: E402
from chokepoint.route_head import RouteHead  # noqa: E402

LEARNER, BEACON = "navigator", "scout"
ROUTE_DIM = 2
WIRE = LATENT_DIM + 2  # anchored, matched width — identical across conditions
# Camera frames for a just-reset env are only guaranteed fresh after a couple
# of rendered steps (the composition eval measured a CHANCE-level decode off
# reset-time frames). The head samples provisionally until this step, then
# freezes the decision the episode is credited to.
DECIDE_STEP = 2


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    # Executor-native regime: route-conditioned shaping ON (cont.pt was trained
    # under it), no abort, no wrong-corridor tax (the head's mistakes must be
    # priced by the HAZARD, which is the physical fact of the world, not by an
    # instructor that already knows the answer).
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
    for p in executor.parameters():
        p.requires_grad_(False)

    if args.condition == "oracle":
        bus = OracleBroadcast(
            comm_radius=args.comm_radius, broadcast_dim=LATENT_DIM, anchored=True
        )
    elif args.condition == "z_t":
        bus = LatentBroadcast(
            encoder, comm_radius=args.comm_radius,
            broadcast_dim=LATENT_DIM, anchored=True,
        )
    else:
        bus = None  # matched-width zeros below

    head = RouteHead(WIRE).to(device)
    optimizer = optim.Adam(head.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"race v8  condition={args.condition}  wire={WIRE}  "
          f"trainable={sum(p.numel() for p in head.parameters()):,}  "
          f"executor FROZEN ({args.executor})")

    def msg_vec() -> torch.Tensor:
        if bus is None:
            return torch.zeros(E, WIRE, device=device)
        messages, mask = bus.deliver(env)[LEARNER]
        return messages[:, 0, :] * mask[:, 0:1].float()

    empty_msg = torch.zeros(E, 0, 1, device=device)
    empty_mask = torch.zeros(E, 0, device=device)
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)

    env.reset()
    # settle rendered frames before the first decode (reset-frame staleness)
    for _ in range(3):
        obs_dict, _, _, _, _ = env.step({LEARNER: zero_scout, BEACON: zero_scout})

    ep_t = torch.zeros(E, dtype=torch.long, device=device)
    ep_ret = torch.zeros(E, device=device)
    acc_haz = torch.zeros(E, device=device)
    choice = torch.zeros(E, dtype=torch.long, device=device)  # 0 top, 1 bottom
    first_corr = torch.zeros(E, dtype=torch.long, device=device)
    dec_msg = torch.zeros(E, WIRE, device=device)
    dec_lp = torch.zeros(E, device=device)
    dec_safe_top = torch.zeros(E, dtype=torch.bool, device=device)

    buf_msg, buf_choice, buf_lp, buf_ret = [], [], [], []
    route_opt = deque(maxlen=500)
    succ = deque(maxlen=500)
    haz = deque(maxlen=500)
    obey = deque(maxlen=500)
    curve = []
    episodes = 0
    updates = 0
    global_step = 0
    start = time.time()

    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv = open(args.log_csv, "w")
        csv.write("episodes,route_opt,success,hazard,obey,mean_return,entropy,sps\n")

    last_entropy = float("nan")
    dec_bad = torch.zeros(E, dtype=torch.bool, device=device)
    n_bad = 0
    while episodes < args.total_episodes:
        # PhysX can NaN a root state when the deterministic executor grinds
        # against a wall for hundreds of steps (stage 1.5 always had sampling
        # noise; v8 does not). A NaN position poisons the camera frame, the
        # message, the shaping reward and finally the head weights — the first
        # launch lost 2/3 conditions this way. Sanitize at the boundary and
        # refuse to train on any episode it touched.
        m = torch.nan_to_num(msg_vec())
        with torch.no_grad():
            logits, _ = head(m)
            dist = Categorical(logits=logits)
            sample = dist.sample()
        deciding = ep_t <= DECIDE_STEP
        choice = torch.where(deciding, sample, choice)
        commit = ep_t == DECIDE_STEP
        if commit.any():
            dec_msg[commit] = m[commit]
            dec_lp[commit] = dist.log_prob(sample)[commit]
            dec_safe_top[commit] = ~env._slab_top[commit]
            dec_bad[commit] = False

        # The env randomizes its own command at reset; the head's choice must
        # drive shaping and the wrong-corridor bookkeeping instead, every step.
        route_top = choice == 0
        env._route_top[:] = route_top
        route = torch.zeros(E, ROUTE_DIM, device=device)
        route[route_top, 0] = 1.0
        route[~route_top, 1] = 1.0

        rgb = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            action = executor.actor(executor.features(rgb, empty_msg, empty_mask, route))
        action = torch.nan_to_num(action)
        obs_dict, rew, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        dec_bad |= ~torch.isfinite(rew[LEARNER])
        global_step += E
        terminated = term[LEARNER]
        done = terminated | tout[LEARNER]
        alive = ~done
        ep_ret += rew[LEARNER]
        acc_haz += env._in_hazard(LEARNER).float() * alive.float()
        in_top = env.in_corridor(LEARNER, top=True) & alive
        in_bot = env.in_corridor(LEARNER, top=False) & alive
        fresh = (first_corr == 0) & (in_top | in_bot)
        first_corr = torch.where(fresh, torch.where(in_top, 1, 2).long(), first_corr)
        ep_t += 1

        if done.any():
            idx = done.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                if bool(dec_bad[i]) or not bool(torch.isfinite(ep_ret[i])) \
                        or not bool(torch.isfinite(dec_msg[i]).all()):
                    n_bad += 1
                    continue
                buf_msg.append(dec_msg[i].clone())
                buf_choice.append(int(choice[i].item()))
                buf_lp.append(float(dec_lp[i].item()))
                buf_ret.append(float(ep_ret[i].item()))
                top_chosen = int(choice[i].item()) == 0
                route_opt.append(float(top_chosen == bool(dec_safe_top[i].item())))
                succ.append(float(terminated[i].item()))
                haz.append(float(acc_haz[i].item()))
                want = 1 if top_chosen else 2
                obey.append(float(int(first_corr[i].item()) == want))
                episodes += 1
            ep_t[idx] = 0
            ep_ret[idx] = 0.0
            acc_haz[idx] = 0.0
            first_corr[idx] = 0
            dec_bad[idx] = False

        if len(buf_ret) >= args.batch_episodes:
            if args.anneal_lr:
                frac = 1.0 - min(1.0, episodes / args.total_episodes)
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            b_msg = torch.stack(buf_msg)
            b_choice = torch.tensor(buf_choice, device=device)
            b_lp = torch.tensor(buf_lp, device=device)
            b_ret = torch.tensor(buf_ret, device=device)
            n = len(buf_ret)
            inds = np.arange(n)
            ent_acc = []
            for _ in range(args.update_epochs):
                np.random.shuffle(inds)
                for s0 in range(0, n, args.minibatch_size):
                    mb = inds[s0 : s0 + args.minibatch_size]
                    # Episodes finish in bursts, so n is rarely an exact
                    # multiple of the minibatch: a 1-element remainder makes
                    # Bessel-corrected std() return NaN, which killed three
                    # runs before being traced here. Population std plus a
                    # size guard closes both holes.
                    if len(mb) < 2:
                        continue
                    logits, v = head(b_msg[mb])
                    d = Categorical(logits=logits)
                    new_lp = d.log_prob(b_choice[mb])
                    ratio = (new_lp - b_lp[mb]).exp()
                    adv = b_ret[mb] - v.detach()
                    adv = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)
                    pg = torch.max(
                        -adv * ratio,
                        -adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    v_loss = 0.5 * ((v - b_ret[mb]) ** 2).mean()
                    ent = d.entropy().mean()
                    ent_acc.append(float(ent))
                    loss = pg - args.ent_coef * ent + args.vf_coef * v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                    optimizer.step()
            last_entropy = float(np.mean(ent_acc))
            buf_msg, buf_choice, buf_lp, buf_ret = [], [], [], []
            updates += 1

            def dm(d):
                return float(np.mean(d)) if d else float("nan")

            sps = int(global_step / (time.time() - start))
            row = dict(
                episodes=episodes, route_opt=dm(route_opt), success=dm(succ),
                hazard=dm(haz), obey=dm(obey), mean_return=float(b_ret.mean()),
                entropy=last_entropy, sps=sps,
            )
            curve.append(row)
            print(f"[v8/{args.condition}] update {updates:3d}  ep {episodes:5d}  "
                  f"route_opt {row['route_opt']:.3f}  succ {row['success']:.3f}  "
                  f"hazard {row['hazard']:5.2f}  obey {row['obey']:.3f}  "
                  f"ret {row['mean_return']:6.2f}  H {row['entropy']:.3f}  "
                  f"bad {n_bad}  SPS {sps}")
            if args.log_csv:
                csv.write(",".join(str(row[k]) for k in [
                    "episodes", "route_opt", "success", "hazard", "obey",
                    "mean_return", "entropy", "sps"]) + "\n")
                csv.flush()

    def dm(d):
        return float(np.mean(d)) if d else float("nan")

    g_opt, g_succ, g_haz = dm(route_opt), dm(succ), dm(haz)
    print(f"\n[v8/{args.condition}/seed{args.seed}] FINAL (last {len(succ)} eps)  "
          f"route_opt {g_opt:.3f}  success {g_succ:.3f}  hazard {g_haz:.2f}  "
          f"obey {dm(obey):.3f}")
    print(f"RACE-V8 READOUT ({args.condition}): route_opt >= 0.90 and hazard <= 2.0: "
          f"{'PASS' if g_opt >= 0.90 and g_haz <= 2.0 else 'FAIL'}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": head.state_dict(), "args": vars(args)}, args.save)
        print(f"saved head -> {args.save}")
    if args.run_json:
        Path(args.run_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.run_json, "w") as f:
            json.dump({"curve": curve, "route_opt": g_opt, "success": g_succ,
                       "hazard": g_haz, "args": vars(args)}, f, indent=2)
        print(f"wrote run json -> {args.run_json}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
