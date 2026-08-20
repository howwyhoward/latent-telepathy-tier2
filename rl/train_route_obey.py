"""Stage 1.5 — train the navigator to OBEY a route command.

Races v1-v7 all failed identically: the policy has no representation of "which
corridor am I taking", so a route preference can only live in the lateral
component of a 600-sample Gaussian, where d logpi/dmu = (a-mu)/sigma^2 makes the
signal that buys coverage the same signal that gets attenuated. v7 proved both
halves of that: far-corridor coverage reached 0.22 and was optimized away by
iteration 80, and the oracle DID recruit the perfect bit — to gate
advance-vs-balk inside the corridor it was already taking (lying to it moves
success 1.00 -> 0.41 without ever moving the corridor).

So the race is split into two problems that are each easy alone:

  stage 1.5 (here)  obey a 1-bit route command
  stage 2           choose the route from the message

Obedience is trained with a DENSE, IMMEDIATE penalty for occupying the corridor
that was not commanded. That is what breaks the chicken-and-egg: a route input
that changes nothing earns no gradient, and in stage 2 a route head whose
choice changes nothing earns none either. The command is set to the SAFE side,
so obeying never fights the hazard and both routes are commanded equally often.

Gate for stage 2: obedience >= 0.9 in BOTH directions.

    python rl/train_route_obey.py --init_nav runs/nav_pretrain/nav_s1_mouth.pt \
        --save runs/route_obey/obey.pt --run_json runs/route_obey/obey.json
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--total_timesteps", type=int, default=3_000_000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=128)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=2.5e-4)
parser.add_argument("--anneal_lr", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--rew_wrong_corridor", type=float, default=-0.25)
parser.add_argument("--spawn_route", type=float, default=0.0,
                    help="fraction of episodes spawned on the commanded route's "
                         "reverse curriculum; the spawn walks from that "
                         "corridor's mouth back to the canonical start")
parser.add_argument("--spawn_route_anneal", type=float, default=0.6,
                    help="fraction of training over which the curriculum spawn "
                         "retreats from the mouth to the canonical start")
parser.add_argument("--spawn_yaw_jitter", type=float, default=0.0,
                    help="uniform heading jitter (rad) on navigator spawns")
parser.add_argument("--spawn_mouths", type=float, default=0.0,
                    help="fraction of episodes starting at the COMMANDED "
                         "corridor mouth facing east, where obeying is already "
                         "achievable; the rest start at the canonical pose")
parser.add_argument("--route_abort_wrong", type=int, default=0,
                    help="end the episode on entering the wrong corridor, "
                         "instead of taxing every step spent in it")
parser.add_argument("--route_shaping", type=int, default=1,
                    help="shape progress via the commanded corridor; 0 leaves "
                         "the plain distance-to-goal field, isolating whether "
                         "the wrong-corridor penalty alone buys obedience")
parser.add_argument("--rew_hazard", type=float, default=-0.05)
parser.add_argument("--rew_hazard_entry", type=float, default=-2.0)
parser.add_argument("--rew_success_obedient_only", type=int, default=0,
                    help="gate the success bonus on never entering the wrong "
                         "corridor, instead of penalizing it everywhere "
                         "(round-5 experiment B)")
parser.add_argument("--init_nav", type=str, default=None,
                    help="stage-1 trunk; actor/critic inputs are zero-widened "
                         "for the route command so the controller loads intact")
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
# v7's exploration schedule, reused: correlated lateral noise for the first
# steps of an episode supplies the far corridor (measured 0.22 coverage for
# 0.51 success vs 0.55 unperturbed). It failed in v7 because the branch it
# produced had no clean advantage attached; here the wrong-corridor penalty
# gives it one that is dense, immediate and uncontaminated.
parser.add_argument("--explore_window", type=int, default=40)
parser.add_argument("--explore_log_std", type=float, default=1.5)
parser.add_argument("--explore_tau", type=float, default=30.0)
parser.add_argument("--explore_anneal_frac", type=float, default=0.5)
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
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402

LEARNER, BEACON = "navigator", "scout"
ROUTE_DIM = 2


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    cfg.route_instruction = True
    cfg.route_shaping = bool(args.route_shaping)
    cfg.route_abort_wrong = bool(args.route_abort_wrong)
    cfg.spawn_mouth_prob = args.spawn_mouths
    cfg.spawn_route_prob = args.spawn_route
    cfg.spawn_route_frac = 1.0
    cfg.spawn_yaw_jitter = args.spawn_yaw_jitter
    cfg.rew_wrong_corridor = args.rew_wrong_corridor
    cfg.rew_hazard = args.rew_hazard
    cfg.rew_hazard_entry = args.rew_hazard_entry
    cfg.rew_success_obedient_only = bool(args.rew_success_obedient_only)
    env = ChokepointEnv(cfg)
    device = env.device

    ck = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ck["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    policy = AttentionReceiver(
        encoder, broadcast_dim=0, latent_dim=LATENT_DIM, route_dim=ROUTE_DIM
    ).to(device)
    if args.init_nav:
        src = torch.load(args.init_nav, map_location=device)["policy"]
        info = policy.load_trunk(src)
        print(f"init_nav: loaded {len(info['loaded'])}/{info['total']} tensors "
              f"from {args.init_nav}; widened {info['widened']}")
    # Actions are scored under the std they were drawn with, which makes the
    # entropy term parameter-free and leaves log_std without a gradient either
    # way. Freeze it explicitly so that is a decision rather than a side effect.
    policy.log_std.requires_grad_(False)
    print(f"log_std frozen at {[round(s, 3) for s in policy.log_std.exp().tolist()]}")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.learning_rate, eps=1e-5)
    print(f"route obedience: trainable={sum(p.numel() for p in trainable):,} "
          f"(encoder frozen)  wrong_corridor={args.rew_wrong_corridor}/step  "
          f"route_shaping={bool(args.route_shaping)}  "
          f"abort_wrong={bool(args.route_abort_wrong)}")

    S, E = args.num_steps, args.num_envs
    obs = torch.zeros((S, E, 3, 64, 64), device=device)
    routes = torch.zeros((S, E, ROUTE_DIM), device=device)
    actions = torch.zeros((S, E, N_ACTIONS), device=device)
    stds = torch.zeros((S, E, N_ACTIONS), device=device)
    logprobs = torch.zeros((S, E), device=device)
    rewards = torch.zeros((S, E), device=device)
    dones = torch.zeros((S, E), device=device)
    values = torch.zeros((S, E), device=device)

    empty_msg = torch.zeros(E, 0, 1, device=device)
    empty_mask = torch.zeros(E, 0, device=device)
    zero_scout = torch.zeros(E, N_ACTIONS, device=device)

    obs_dict, _ = env.reset()
    next_obs = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
    next_route = env.route_onehot()
    next_done = torch.zeros(E, device=device)

    rho = float(np.exp(-1.0 / args.explore_tau)) if args.explore_tau > 0 else 0.0
    eps = torch.randn(E, N_ACTIONS, device=device)
    ep_t = torch.zeros(E, device=device)

    # Obedience is scored on the FIRST corridor entered, matching the cross-tab
    # in spike/diag_route_choice.py: a policy that wanders into the wrong
    # corridor and corrects has not obeyed.
    first_corr = torch.zeros(E, dtype=torch.long, device=device)  # 0 none,1 top,2 bot
    ep_cmd_top = env._route_top.clone()
    ep_mouth = env.curriculum_spawn.clone()
    acc_wrong = torch.zeros(E, device=device)

    obey = {1: deque(maxlen=200), 0: deque(maxlen=200)}
    succ = {1: deque(maxlen=200), 0: deque(maxlen=200)}
    # A mouth spawn starts inside the corridor it was told to use, so obeying is
    # already most of the way done. Blending the two spawn types hides whether
    # anything was learned about the canonical start, which is the actual task,
    # so the gate is judged on canonical episodes only.
    obey_can = {1: deque(maxlen=200), 0: deque(maxlen=200)}
    succ_can = {1: deque(maxlen=200), 0: deque(maxlen=200)}
    wrong_steps: deque = deque(maxlen=200)
    curve = []
    best_joint = -1.0
    global_step = 0
    start_time = time.time()

    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv = open(args.log_csv, "w")
        csv.write("global_step,obey_top,obey_bottom,success_top,success_bottom,"
                  "obey_top_can,obey_bottom_can,success_top_can,"
                  "success_bottom_can,wrong_steps,segment_return,approx_kl,sps\n")

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        if args.explore_window > 0:
            a = args.explore_anneal_frac
            k = min(1.0, (iteration - 1) / (a * args.num_iterations)) if a > 0 else 1.0
            boost_std = float(np.exp(
                (1 - k) * args.explore_log_std + k * float(policy.log_std[1])
            ))
        if args.spawn_route > 0.0:
            # Retreat the curriculum spawn from the commanded mouth back to the
            # canonical start. A fixed mouth spawn saturates at ~1.00 obedience
            # there while the canonical start stays at 0.00, so the spawn has to
            # traverse the gap rather than sit at one end of it.
            a = args.spawn_route_anneal
            k = min(1.0, (iteration - 1) / (a * args.num_iterations)) if a > 0 else 1.0
            env.cfg.spawn_route_frac = float(1.0 - k)

        ep_returns: list = []
        for step in range(S):
            global_step += E
            obs[step] = next_obs
            routes[step] = next_route
            dones[step] = next_done

            with torch.no_grad():
                h = policy.features(next_obs, empty_msg, empty_mask, next_route)
                mean, value = policy.actor(h), policy.critic(h)
                std = policy.log_std.exp().expand(E, N_ACTIONS).clone()
                if args.explore_window > 0:
                    std[ep_t < args.explore_window, 1:2] = boost_std
                eps = (
                    rho * eps + (1.0 - rho**2) ** 0.5 * torch.randn_like(eps)
                    if rho > 0.0
                    else torch.randn_like(eps)
                )
                action = mean + std * eps
                logprob = Normal(mean, std).log_prob(action).sum(-1)
            values[step] = value.flatten()
            actions[step] = action
            stds[step] = std
            logprobs[step] = logprob
            ep_t += 1

            obs_dict, rew, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            r = rew[LEARNER]
            terminated = term[LEARNER]
            done = (terminated | tout[LEARNER]).float()
            rewards[step] = r
            next_obs = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
            next_route = env.route_onehot()
            next_done = done

            # env.step auto-resets terminated envs, so their pose already belongs
            # to the NEXT episode; exclude them or the tallies leak across it.
            alive = done == 0
            in_top = env.in_corridor(LEARNER, top=True) & alive
            in_bot = env.in_corridor(LEARNER, top=False) & alive
            fresh = (first_corr == 0) & (in_top | in_bot)
            first_corr = torch.where(fresh, torch.where(in_top, 1, 2).long(), first_corr)
            acc_wrong += env.in_wrong_corridor(LEARNER).float() * alive.float()

            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    cmd = int(ep_cmd_top[i].item())          # 1 top, 0 bottom
                    want = 1 if cmd else 2
                    obeyed = float(int(first_corr[i].item()) == want)
                    arrived = float(terminated[i].item())
                    obey[cmd].append(obeyed)
                    succ[cmd].append(arrived)
                    if not bool(ep_mouth[i].item()):
                        obey_can[cmd].append(obeyed)
                        succ_can[cmd].append(arrived)
                    wrong_steps.append(acc_wrong[i].item())
                first_corr[idx] = 0
                acc_wrong[idx] = 0.0
                eps[idx] = torch.randn_like(eps[idx])
                ep_t[idx] = 0
                ep_cmd_top = env._route_top.clone()
                ep_mouth = env.curriculum_spawn.clone()
            ep_returns.append(float(r.mean()))

        with torch.no_grad():
            next_value = policy.get_value(
                next_obs, empty_msg, empty_mask, next_route
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(S)):
                if t == S - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape(-1, 3, 64, 64)
        b_routes = routes.reshape(-1, ROUTE_DIM)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, N_ACTIONS)
        b_stds = stds.reshape(-1, N_ACTIONS)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(args.batch_size)
        approx_kl = torch.tensor(0.0)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                mb_msg = torch.zeros(len(mb), 0, 1, device=device)
                mb_mask = torch.zeros(len(mb), 0, device=device)
                # score each action under the std it was drawn with, or the
                # window's wide samples fall in the tail and get clipped away
                h = policy.features(b_obs[mb], mb_msg, mb_mask, b_routes[mb])
                dist = Normal(policy.actor(h), b_stds[mb])
                newlogprob = dist.log_prob(b_actions[mb]).sum(-1)
                entropy = dist.entropy().sum(-1)
                newvalue = policy.critic(h)

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                v_clipped = b_values[mb] + torch.clamp(
                    newvalue - b_values[mb], -args.clip_coef, args.clip_coef
                )
                v_loss = 0.5 * torch.max(
                    (newvalue - b_returns[mb]) ** 2,
                    (v_clipped - b_returns[mb]) ** 2,
                ).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()

        def m(d):
            return float(np.mean(d)) if d else float("nan")

        sps = int(global_step / (time.time() - start_time))
        row = dict(
            global_step=global_step, obey_top=m(obey[1]), obey_bottom=m(obey[0]),
            success_top=m(succ[1]), success_bottom=m(succ[0]),
            obey_top_can=m(obey_can[1]), obey_bottom_can=m(obey_can[0]),
            success_top_can=m(succ_can[1]), success_bottom_can=m(succ_can[0]),
            wrong_steps=m(wrong_steps), segment_return=float(np.sum(ep_returns)),
            approx_kl=float(approx_kl), sps=sps,
        )
        curve.append(row)
        print(f"[obey] iter {iteration:3d}/{args.num_iterations}  "
              f"step {global_step:8d}  obey top {row['obey_top']:.2f}/"
              f"bot {row['obey_bottom']:.2f}  succ top {row['success_top']:.2f}/"
              f"bot {row['success_bottom']:.2f}  | canonical obey "
              f"{row['obey_top_can']:.2f}/{row['obey_bottom_can']:.2f} succ "
              f"{row['success_top_can']:.2f}/{row['success_bottom_can']:.2f}"
              f"  approx_kl {row['approx_kl']:.4f}  SPS {sps}")
        if args.log_csv:
            csv.write(",".join(str(row[k]) for k in [
                "global_step", "obey_top", "obey_bottom", "success_top",
                "success_bottom", "obey_top_can", "obey_bottom_can",
                "success_top_can", "success_bottom_can",
                "wrong_steps", "segment_return", "approx_kl", "sps",
            ]) + "\n")
            csv.flush()

        # Peak snapshotting (19 Aug audit): every realcam20 run peaked mid-run
        # and declined -- 0 of ~5000 end-of-run points beat the mid-run bests
        # on both axes -- so end-of-run selection threw the best policies away.
        # Joint = worst-direction obedience x worst-direction success, smoothed.
        if args.save and len(curve) >= 9:
            w9 = curve[-9:]
            jo = min(float(np.nanmean([r["obey_top"] for r in w9])),
                     float(np.nanmean([r["obey_bottom"] for r in w9])))
            js = min(float(np.nanmean([r["success_top"] for r in w9])),
                     float(np.nanmean([r["success_bottom"] for r in w9])))
            joint = jo * js
            if joint > best_joint:
                best_joint = joint
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                tmp = args.save + ".best.tmp"
                torch.save({"policy": policy.state_dict(), "args": vars(args),
                            "iteration": iteration, "global_step": global_step,
                            "joint_obey": jo, "joint_succ": js}, tmp)
                Path(tmp).rename(args.save + ".best")
                print(f"[obey] new joint peak {joint:.4f} "
                      f"(obey {jo:.3f} succ {js:.3f}) -> {args.save}.best")

        # crash insurance: two machine outages have eaten full 3M-step runs
        if args.save and iteration % 25 == 0:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            tmp = args.save + ".latest.tmp"
            torch.save({"policy": policy.state_dict(), "args": vars(args),
                        "iteration": iteration, "global_step": global_step}, tmp)
            Path(tmp).rename(args.save + ".latest")

    # Obedience alone is gameable and was gamed: it scores the FIRST corridor
    # entered, so entering the commanded one and parking read 0.93 obedience at
    # 0.000 success (v2). Stage 2 swaps the command for the scout's message, so
    # a route the policy cannot execute leaves that stage nothing to build on --
    # the gate has to demand arrival too.
    # canonical-only: a mouth spawn already sits in the commanded corridor
    g_obey = min(m(obey_can[1]), m(obey_can[0]))
    g_succ = min(m(succ_can[1]), m(succ_can[0]))
    print(f"\n[obey/seed{args.seed}] FINAL obey top {m(obey[1]):.3f} "
          f"bottom {m(obey[0]):.3f}  success top {m(succ[1]):.3f} "
          f"bottom {m(succ[0]):.3f}   (all spawns)")
    print(f"[obey/seed{args.seed}] CANONICAL obey top {m(obey_can[1]):.3f} "
          f"bottom {m(obey_can[0]):.3f}  success top {m(succ_can[1]):.3f} "
          f"bottom {m(succ_can[0]):.3f}")
    print(f"STAGE-2 GATE (canonical spawns; obedience >= 0.90 AND success >= 0.80, both ways): "
          f"{'PASS' if g_obey >= 0.90 and g_succ >= 0.80 else 'FAIL'} "
          f"(worst obedience {g_obey:.3f}, worst success {g_succ:.3f})")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": policy.state_dict(), "args": vars(args)}, args.save)
        print(f"saved policy -> {args.save}")
    if args.run_json:
        Path(args.run_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.run_json, "w") as f:
            json.dump({"curve": curve, "gate_obey": g_obey, "gate_succ": g_succ,
                       "args": vars(args)}, f, indent=2)
        print(f"wrote run json -> {args.run_json}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
