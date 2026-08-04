"""Phase 3: the reduced race (Tier 2's M10c on pixels).

One condition per invocation. Tier 1's decisive chokepoint design carries
over exactly:

  - The SCOUT is a static beacon: parked at its start pose (facing the top
    corridor, so it sees the slab iff the slab is in the top corridor),
    excluded from the PPO update. Its camera feeds the message bus.
  - The NAVIGATOR is the only learner: an AttentionReceiver over the FROZEN
    Phase 2 JEPA encoder (perception identical across conditions), pooling
    the scout's message via masked attention with the zero-init value path
    (every condition starts as the `none` policy).
  - Conditions differ ONLY in `get_broadcast_content`:
      none     : no channel (floor)
      position : anchor + zeros at matched 64-D width (what does content add
                 beyond position-sharing?)
      z_t      : anchor + frozen JEPA latent of the scout's view (the thesis)
      raw      : anchor + the scout's full 64x64x3 frame (12288-D ceiling)
  - Pre-registered readout: hazard-steps/episode. The no-comms floor from M7
    is ~21 (blind 50/50 corridor pick). A useful message lets the navigator
    pick the clean corridor: hazard-steps -> ~0 while success stays >= none.

PPO core is identical to rl/ppo_pixels.py (which passed M7).

Run (one condition):
    python rl/train_race.py --condition z_t --seed 1 \
        --run_json runs/race/z_t_s1.json
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default="z_t",
                    choices=["none", "position", "z_t", "raw", "oracle"])
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0,
                    help="always-in-range for the race (content, not range, is the contrast)")
parser.add_argument("--total_timesteps", type=int, default=3_000_000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=128)
# Discount must be re-derived under Tier 1 -> Tier 2 time rescaling. Tier 1:
# one decision = one grid step, corridor choice paid off ~20 steps later, so
# gamma 0.99 gave it 0.82 credit. Here one cell of motion is ~10 control
# steps, so the same choice pays off ~240 steps later — gamma 0.99 discounts
# it to 0.09 and the decision is nearly invisible to the optimizer. Matching
# the effective horizon wants gamma ~ 0.99^(1/10) = 0.999.
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--clip_coef", type=float, default=0.2)
# v3 economics. History of the knife edge:
#   v1 (-0.05/step, ent 0): crossing ~-2 total, too cheap — every condition
#      blind-commits and ignores the message (diag: probe 1.00, ablation null).
#   v2 (-0.15/step, ent .01): crossing = a 43-step penalty VALLEY — every
#      condition refuses to cross and times out on the slab side (succ ~0.4).
# v3 keeps the light per-step term but moves the sting into a one-time ENTRY
# cost: the entry penalty is sunk the moment it lands, so the gradient past
# the edge points forward and refusing never beats crossing.
#   v3 (entry -3.0, rung open): no refusal for z_t, but the open rung let a
#      BLIND policy peek-and-reroute for ~0.7 return — message worth ~nothing
#      (ablation null). Also `none` still refused at the -3 edge.
# v4: the rung is sealed in the scene (geometry.remove_rung) so a wrong
# corridor costs a crossing or a full backtrack; entry softened to -2.0
# (crossing ~-4.2 total vs +10 success) to keep refusal strictly dominated.
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--rew_hazard", type=float, default=-0.05,
                    help="per-step hazard penalty inside the slab")
parser.add_argument("--rew_hazard_entry", type=float, default=-2.0,
                    help="one-time penalty on slab entry (rising edge)")
# v5: corridor choice is bistable under Gaussian action noise (v4: runs froze
# into blind-crossing OR refusal by early luck, kl->0, message never
# recruited). Random navigator spawns make both corridors trained territory;
# all race metrics below are computed on canonical-start episodes ONLY.
parser.add_argument("--spawn_curriculum", type=float, default=0.5,
                    help="prob. an episode spawns the navigator at a random free pose")
# v6: two-stage training. Mixing random spawns INTO the race (v5) halved the
# on-task data and stalled 3/4 conditions at ~0.05 success. Instead: stage 1
# pretrains pure navigation (condition none, spawn_curriculum 1.0) until the
# goal is reachable from anywhere; stage 2 initializes EVERY condition from
# that same trunk (ego_proj/actor/critic/log_std; the message branch stays at
# its zero-init, so all conditions still start as the none policy) and races
# on canonical starts. Corridor choice becomes a decision between two
# already-competent routes — the gradient the message needs, without paying
# navigation-learning costs inside the race.
parser.add_argument("--init_nav", type=str, default=None,
                    help="stage-1 checkpoint: warm-start the shared trunk from it")
# Reverse curriculum for stage 1: curriculum spawns restricted to a geodesic
# distance band (0, d_hi) from the goal, with d_hi annealed 2 m -> this value
# over training. Uniform spawning taught nothing (east chamber 0.73, corridor
# regions 0.00): near-goal spawns are trivial, far spawns hopeless, and the
# frontier in between never got dense reward. 0 disables.
parser.add_argument("--spawn_band_max", type=float, default=0.0,
                    help="stage-1 reverse curriculum: final geodesic band max (m)")
# Stage-1 v3: the only spawn curriculum that matches how this policy class
# actually learns (fixed pose + fixed heading, cf. v4 reaching 0.99 from the
# canonical start while every random-yaw curriculum flatlined). Mouth spawns
# force BOTH corridors to be traversed and valued; stationary mixture, no
# annealing, no forgetting frontier.
parser.add_argument("--spawn_mouths", type=float, default=0.0,
                    help="prob. an episode spawns the navigator at a corridor mouth, east-facing")
# v7: v6 was a clean null — an oracle slab bit scored 0.44/0.53 against 0.50
# for silence, at either discount. The precondition was never met: reaching
# the far mouth needs ~+1.3 of SUSTAINED lateral action across the ~30-step
# decision window, and under iid noise the mean deviation over those steps has
# std sigma/sqrt(30) ~ 0.09 — a 14-sigma event, so the alternative corridor is
# never in the batch and there is nothing for a message to be recruited by.
# (Tier 1 never hit this: one gridworld action moved a whole cell.) Fix is
# correlation, not scale: AR(1) noise with rho=exp(-1/tau) has marginal
# N(0,1) at every step, so per-step log-probs and PPO's importance ratios stay
# exact while deviations persist long enough to reach the other corridor.
parser.add_argument("--explore_tau", type=float, default=0.0,
                    help="exploration noise correlation time in steps (0 = iid)")
# Correlated noise across the WHOLE episode buys coverage but destroys the run
# it is measured on (diag_exploration: top 0.00 -> 0.16, success 0.68 -> 0.11 —
# 600 steps of wide noise is 600 steps of driving into walls). The corridor is
# chosen in the first ~3 s and the choice is purely lateral, so the boost is
# confined to a window at episode start and (optionally) to the vy axis; after
# the window the trained noise returns and the mouth-competent trunk executes
# the route cleanly, so the return reflects the ROUTE, not the noise.
parser.add_argument("--explore_window", type=int, default=0,
                    help="steps from episode start with boosted noise (0 = all)")
parser.add_argument("--explore_log_std", type=float, default=0.5,
                    help="log_std applied inside the exploration window")
parser.add_argument("--explore_dims", type=str, default="y", choices=["all", "y"],
                    help="axes to boost: all, or just body-frame lateral (vy)")
# The boost must also go away. At std 4.48 the sample swamps the mean, so the
# policy can reach at most P(top) = Phi(1/4.48) ~ 0.59 however sure it is —
# useful for coverage, useless for committing, and it would poison the headline
# number. Decay it to the base std over the first anneal_frac of training: the
# wide phase supplies the branch, the clean tail lets the mean own the choice
# and makes the final metrics on-policy.
parser.add_argument("--explore_anneal_frac", type=float, default=0.6,
                    help="fraction of training over which the boost decays to base")
parser.add_argument("--log_std_init", type=float, default=None,
                    help="override the warm-started log_std at stage-2 start")
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=2.5e-4)
parser.add_argument("--anneal_lr", type=int, default=1)
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
from torch.distributions import Normal

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

LEARNER = "navigator"
BEACON = "scout"


def make_bus(condition: str, encoder) -> MessageBus:
    """Tier 1 semantics: every message condition is anchored; the matched
    channel width is 64 floats; `none` is the channel-less floor."""
    if condition == "none":
        return MessageBus(comm_radius=args.comm_radius, broadcast_dim=0)
    if condition == "position":
        # anchor + zero content at matched width == literally position-only
        return MessageBus(
            comm_radius=args.comm_radius, broadcast_dim=LATENT_DIM, anchored=True
        )
    if condition == "z_t":
        return LatentBroadcast(
            encoder, comm_radius=args.comm_radius,
            broadcast_dim=LATENT_DIM, anchored=True,
        )
    if condition == "raw":
        return RawObsBroadcast(comm_radius=args.comm_radius, anchored=True)
    if condition == "oracle":
        return OracleBroadcast(
            comm_radius=args.comm_radius, broadcast_dim=LATENT_DIM, anchored=True
        )
    raise ValueError(condition)


def main():
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    cfg.rew_hazard = args.rew_hazard
    cfg.rew_hazard_entry = args.rew_hazard_entry
    cfg.spawn_curriculum_prob = args.spawn_curriculum
    cfg.spawn_mouth_prob = args.spawn_mouths
    env = ChokepointEnv(cfg)
    device = env.device
    obs_dict, _ = env.reset()

    # frozen Phase 2 encoder: shared ego perception AND the z_t sender
    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    bus = make_bus(args.condition, encoder)
    wire = bus.wire_dim
    policy = AttentionReceiver(
        encoder, broadcast_dim=wire, latent_dim=LATENT_DIM
    ).to(device)
    if args.init_nav:
        # Warm-start the shared trunk from the stage-1 navigation policy.
        # Message-branch keys (msg_proj/q/k/v) are absent from a `none`
        # source and keep their fresh zero-init value path.
        src = torch.load(args.init_nav, map_location=device)["policy"]
        own = policy.state_dict()
        hits = {
            k: v for k, v in src.items()
            if k in own and own[k].shape == v.shape
        }
        own.update(hits)
        policy.load_state_dict(own)
        print(f"init_nav: loaded {len(hits)}/{len(own)} tensors from {args.init_nav}")

    if args.log_std_init is not None:
        with torch.no_grad():
            policy.log_std.fill_(args.log_std_init)
    if args.explore_window > 0:
        # Exploration is now a schedule, not a learned quantity: scoring actions
        # under their behaviour std makes the entropy term parameter-free, and a
        # log_std free to collapse would quietly retire the schedule's purpose.
        policy.log_std.requires_grad_(False)
        print(f"log_std frozen at {policy.log_std.exp().tolist()}; "
              f"window {args.explore_window} steps @ std "
              f"{float(np.exp(args.explore_log_std)):.2f} on {args.explore_dims}")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.learning_rate, eps=1e-5)
    print(
        f"condition={args.condition}  wire_dim={wire}  "
        f"trainable={sum(p.numel() for p in trainable):,} (encoder frozen)"
    )

    zero_scout = torch.zeros(args.num_envs, N_ACTIONS, device=device)

    def nav_inputs(o):
        rgb = o[LEARNER].permute(0, 3, 1, 2).contiguous()
        msgs, mask = bus.deliver(env)[LEARNER]
        return rgb, msgs, mask.float()

    S, E = args.num_steps, args.num_envs
    obs = torch.zeros((S, E, 3, 64, 64), device=device)
    msgs = torch.zeros((S, E, 1, wire), device=device)
    masks = torch.zeros((S, E, 1), device=device)
    actions = torch.zeros((S, E, N_ACTIONS), device=device)
    stds = torch.zeros((S, E, N_ACTIONS), device=device)
    logprobs = torch.zeros((S, E), device=device)
    rewards = torch.zeros((S, E), device=device)
    dones = torch.zeros((S, E), device=device)
    values = torch.zeros((S, E), device=device)

    next_obs, next_msg, next_mask = nav_inputs(obs_dict)
    next_done = torch.zeros(E, device=device)

    rho = float(np.exp(-1.0 / args.explore_tau)) if args.explore_tau > 0 else 0.0
    eps = torch.randn(E, N_ACTIONS, device=device)
    ep_t = torch.zeros(E, device=device)
    boost_axes = slice(None) if args.explore_dims == "all" else slice(1, 2)

    acc_ret = torch.zeros(E, device=device)
    acc_len = torch.zeros(E, device=device)
    acc_hazard = torch.zeros(E, device=device)
    # headline metrics: canonical-start episodes only (the race task).
    # Curriculum (random-spawn) episodes are training fuel; their success is
    # tracked separately as a skill-coverage signal.
    ep_returns: deque = deque(maxlen=100)
    ep_lengths: deque = deque(maxlen=100)
    ep_successes: deque = deque(maxlen=100)
    ep_hazards: deque = deque(maxlen=100)
    ep_succ_slab_top: deque = deque(maxlen=100)
    ep_succ_slab_bot: deque = deque(maxlen=100)
    ep_succ_rand: deque = deque(maxlen=100)
    ep_slab = env._slab_top.clone()
    ep_rand = env.curriculum_spawn.clone()
    curve = []

    csv_file = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w")
        csv_file.write(
            "global_step,success,success_slab_top,success_slab_bottom,"
            "ep_len,mean_return,hazard_steps,approx_kl,sps,success_rand_spawn\n"
        )

    global_step = 0
    start_time = time.time()

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
        if args.spawn_band_max > 0:
            d_hi = max(2.0, args.spawn_band_max * iteration / args.num_iterations)
            env.cfg.spawn_dist_range = (0.0, d_hi)

        for step in range(S):
            global_step += E
            obs[step] = next_obs
            msgs[step] = next_msg
            masks[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                h = policy.features(next_obs, next_msg, next_mask)
                mean, value = policy.actor(h), policy.critic(h)
                std = policy.log_std.exp().expand(E, N_ACTIONS).clone()
                if args.explore_window > 0:
                    hot = ep_t < args.explore_window
                    std[hot, boost_axes] = boost_std
                eps = (
                    rho * eps + (1.0 - rho**2) ** 0.5 * torch.randn_like(eps)
                    if rho > 0.0
                    else torch.randn_like(eps)
                )
                action = mean + std * eps
                logprob = Normal(mean, std).log_prob(action).sum(-1)
            values[step] = value.flatten()
            stds[step] = std
            ep_t += 1
            actions[step] = action
            logprobs[step] = logprob

            obs_dict, rew, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
            )
            r = rew[LEARNER]
            terminated = term[LEARNER]
            done = (terminated | tout[LEARNER]).float()

            rewards[step] = r
            next_obs, next_msg, next_mask = nav_inputs(obs_dict)
            next_done = done

            acc_ret += r
            acc_len += 1
            acc_hazard += env._in_hazard(LEARNER).float() * (1.0 - done)
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                eps[idx] = torch.randn_like(eps[idx])
                ep_t[idx] = 0
                for i in idx.tolist():
                    succ = float(terminated[i].item())
                    if ep_rand[i]:
                        ep_succ_rand.append(succ)
                    else:
                        ep_returns.append(acc_ret[i].item())
                        ep_lengths.append(acc_len[i].item())
                        ep_successes.append(succ)
                        ep_hazards.append(acc_hazard[i].item())
                        (ep_succ_slab_top if ep_slab[i] else ep_succ_slab_bot).append(succ)
                acc_ret[idx] = 0.0
                acc_len[idx] = 0.0
                acc_hazard[idx] = 0.0
                ep_slab[idx] = env._slab_top[idx]
                ep_rand[idx] = env.curriculum_spawn[idx]

        # GAE
        with torch.no_grad():
            next_value = policy.get_value(next_obs, next_msg, next_mask).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(S)):
                if t == S - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # explicit batch size: wire==0 (the `none` floor) makes a -1 reshape
        # ambiguous (zero total elements) — same gotcha as Tier 1
        B = S * E
        b_obs = obs.reshape(B, 3, 64, 64)
        b_msgs = msgs.reshape(B, 1, wire)
        b_masks = masks.reshape(B, 1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, N_ACTIONS)
        b_stds = stds.reshape(-1, N_ACTIONS)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                # Score each action under the std it was actually drawn with.
                # Re-scoring a window action under the (much tighter) learned
                # log_std would put it far in the tail, drive its ratio to ~0
                # and clip away the exploratory trajectories we paid for.
                h = policy.features(b_obs[mb], b_msgs[mb], b_masks[mb])
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

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(
                    newvalue - b_values[mb], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
        succ = np.mean(ep_successes) if ep_successes else float("nan")
        mean_len = np.mean(ep_lengths) if ep_lengths else float("nan")
        mean_haz = np.mean(ep_hazards) if ep_hazards else float("nan")
        s_top = np.mean(ep_succ_slab_top) if ep_succ_slab_top else float("nan")
        s_bot = np.mean(ep_succ_slab_bot) if ep_succ_slab_bot else float("nan")
        s_rand = np.mean(ep_succ_rand) if ep_succ_rand else float("nan")
        curve.append((global_step, float(succ), float(mean_ret), float(mean_len), float(mean_haz)))
        print(
            f"[{args.condition}] iter {iteration:3d}/{args.num_iterations}  "
            f"step {global_step:>8d}  return {mean_ret:7.3f}  "
            f"success {succ:5.2f} (top {s_top:4.2f}/bot {s_bot:4.2f})  "
            f"rand_succ {s_rand:4.2f}  "
            f"ep_len {mean_len:6.1f}  hazard_steps {mean_haz:5.1f}  "
            f"approx_kl {approx_kl.item():.4f}  SPS {sps}",
            flush=True,
        )
        if csv_file is not None:
            csv_file.write(
                f"{global_step},{succ},{s_top},{s_bot},{mean_len},{mean_ret},"
                f"{mean_haz},{approx_kl.item()},{sps},{s_rand}\n"
            )
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    final_succ = float(np.mean(ep_successes)) if ep_successes else 0.0
    final_haz = float(np.mean(ep_hazards)) if ep_hazards else float("nan")
    final_rand = float(np.mean(ep_succ_rand)) if ep_succ_rand else float("nan")
    print(f"\n[{args.condition}/seed{args.seed}] FINAL success {final_succ:.3f}  "
          f"hazard_steps {final_haz:.2f}  rand_spawn_success {final_rand:.3f}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"policy": policy.state_dict(), "condition": args.condition,
             "seed": args.seed},
            args.save,
        )
        print(f"saved policy -> {args.save}")
    if args.run_json:
        Path(args.run_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.run_json, "w") as f:
            json.dump(
                {
                    "condition": args.condition,
                    "seed": args.seed,
                    "final_success": final_succ,
                    "final_hazard_steps": final_haz,
                    "success_slab_top": float(np.mean(ep_succ_slab_top)) if ep_succ_slab_top else None,
                    "success_slab_bottom": float(np.mean(ep_succ_slab_bot)) if ep_succ_slab_bot else None,
                    "curve": curve,
                    "total_timesteps": args.total_timesteps,
                    "comm_radius": args.comm_radius,
                    "rew_hazard": args.rew_hazard,
                    "rew_hazard_entry": args.rew_hazard_entry,
                    "ent_coef": args.ent_coef,
                    "spawn_curriculum": args.spawn_curriculum,
                    "success_rand_spawn": float(np.mean(ep_succ_rand)) if ep_succ_rand else None,
                    "init_nav": args.init_nav,
                    "explore_tau": args.explore_tau,
                    "explore_window": args.explore_window,
                    "explore_log_std": args.explore_log_std,
                    "explore_dims": args.explore_dims,
                    "explore_anneal_frac": args.explore_anneal_frac,
                    "log_std_init": args.log_std_init,
                },
                f,
                indent=2,
            )
        print(f"wrote run json -> {args.run_json}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
