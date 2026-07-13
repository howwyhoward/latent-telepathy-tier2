"""Single-agent PPO positive control on pixels (Tier 2's M7).

Faithful port of Tier 1's rl/ppo_smoke.py: same CleanRL-style core (orthogonal
init, GAE, advantage normalization, clipped policy + value loss, minibatch
epochs, LR anneal), with three deliberate changes for Isaac:

  - continuous diagonal-Gaussian actions over cmd_vel (vx, vy, wz) instead of
    Categorical over grid moves
  - a small trainable CNN over 64x64 RGB instead of an MLP over one-hot
    patches (the frozen JEPA encoder arrives in Phase 2; M7's job is to prove
    the RL/env plumbing learns AT ALL, so the encoder trains end to end)
  - the env is natively vectorized on GPU, so the per-env Python loop is gone

Task easing, mirroring Tier 1's "small map + large FOV": only the NAVIGATOR
learns and only its goal terminates the episode (cfg.success_agents); the
scout is parked with zero actions. Dense potential-based progress shaping is
already in the env reward.

The navigator cannot see which corridor holds the hazard slab (that is the
certified occlusion), so the expected outcome is success >= 0.8 with roughly
half of episodes eating hazard penalty. That hazard rate is the no-comms
floor every message condition must beat in Phase 3 - it is logged per episode.

Run (inside tmux; ~20-40 min at 64 envs):

    source setup/env.sh
    python rl/ppo_pixels.py --num_envs 64 --total_timesteps 2000000 \
        --log_csv runs/m7.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--total_timesteps", type=int, default=2_000_000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=128, help="rollout length per iteration")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--clip_coef", type=float, default=0.2)
# 0.0 is the CleanRL default for continuous control: the Gaussian's own std
# (log_std parameter) provides exploration; an entropy bonus mostly inflates it.
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=2.5e-4)
parser.add_argument("--anneal_lr", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_csv", type=str, default=None)
parser.add_argument("--save", type=str, default=None, help="path to save final agent state_dict")
parser.add_argument("--init_from", type=str, default=None,
                    help="warm-start from a saved state_dict (fresh optimizer/LR schedule)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import traceback
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def _die_loudly(exc_type, exc, tb):
    # Kit's teardown deadlocks after an exception in headless mode; exit hard
    # so a crash fails in seconds instead of hanging silently.
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die_loudly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.agent import LEARNER, Agent  # noqa: E402
from chokepoint.constants import N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402


def main():
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device
    obs_dict, _ = env.reset()

    def to_chw(o):
        return o[LEARNER].permute(0, 3, 1, 2).contiguous()

    agent = Agent().to(device)
    if args.init_from:
        agent.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"warm-started from {args.init_from}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    zero_scout = torch.zeros(args.num_envs, N_ACTIONS, device=device)

    # rollout storage (obs at 64x64x3 float32: ~200 MB for 128x64 - fine)
    obs = torch.zeros((args.num_steps, args.num_envs, 3, 64, 64), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, N_ACTIONS), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs = to_chw(obs_dict)
    next_done = torch.zeros(args.num_envs, device=device)

    # per-env episode accumulators (env auto-resets, so we track our own stats)
    acc_ret = torch.zeros(args.num_envs, device=device)
    acc_len = torch.zeros(args.num_envs, device=device)
    acc_hazard = torch.zeros(args.num_envs, device=device)
    ep_returns: deque = deque(maxlen=100)
    ep_lengths: deque = deque(maxlen=100)
    ep_successes: deque = deque(maxlen=100)
    ep_hazards: deque = deque(maxlen=100)
    # success split by where the slab was THIS episode (env re-flips it at
    # reset, so snapshot the side while the episode is running)
    ep_succ_slab_top: deque = deque(maxlen=100)
    ep_succ_slab_bot: deque = deque(maxlen=100)
    ep_slab = env._slab_top.clone()

    csv_file = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w")
        csv_file.write(
            "global_step,success,success_slab_top,success_slab_bottom,"
            "ep_len,mean_return,hazard_steps,approx_kl,sps\n"
        )

    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            obs_dict, rew, term, tout, _ = env.step(
                {LEARNER: action.clamp(-1, 1), "scout": zero_scout}
            )
            r = rew[LEARNER]
            terminated = term[LEARNER]
            timed_out = tout[LEARNER]
            done = (terminated | timed_out).float()

            rewards[step] = r
            next_obs = to_chw(obs_dict)
            next_done = done

            # episode bookkeeping (hazard check runs on the post-step pose,
            # which for done envs is already reset - mask those out)
            acc_ret += r
            acc_len += 1
            acc_hazard += env._in_hazard(LEARNER).float() * (1.0 - done)
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    ep_returns.append(acc_ret[i].item())
                    ep_lengths.append(acc_len[i].item())
                    succ = float(terminated[i].item())
                    ep_successes.append(succ)
                    ep_hazards.append(acc_hazard[i].item())
                    (ep_succ_slab_top if ep_slab[i] else ep_succ_slab_bot).append(succ)
                acc_ret[idx] = 0.0
                acc_len[idx] = 0.0
                acc_hazard[idx] = 0.0
                ep_slab[idx] = env._slab_top[idx]  # side for the episode just started

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
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

        b_obs = obs.reshape(-1, 3, 64, 64)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, N_ACTIONS)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
        succ = np.mean(ep_successes) if ep_successes else float("nan")
        mean_len = np.mean(ep_lengths) if ep_lengths else float("nan")
        mean_haz = np.mean(ep_hazards) if ep_hazards else float("nan")
        s_top = np.mean(ep_succ_slab_top) if ep_succ_slab_top else float("nan")
        s_bot = np.mean(ep_succ_slab_bot) if ep_succ_slab_bot else float("nan")
        print(
            f"iter {iteration:3d}/{args.num_iterations}  step {global_step:>8d}  "
            f"return {mean_ret:7.3f}  success {succ:5.2f} (top {s_top:4.2f}/bot {s_bot:4.2f})  "
            f"ep_len {mean_len:6.1f}  hazard_steps {mean_haz:5.1f}  "
            f"approx_kl {approx_kl.item():.4f}  SPS {sps}",
            flush=True,
        )
        if csv_file is not None:
            csv_file.write(
                f"{global_step},{succ},{s_top},{s_bot},{mean_len},{mean_ret},{mean_haz},"
                f"{approx_kl.item()},{sps}\n"
            )
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), args.save)
        print(f"saved agent to {args.save}")

    final_succ = np.mean(ep_successes) if ep_successes else 0.0
    print(f"\nFINAL success rate (last 100 eps): {final_succ:.2f}")
    print("M7 POSITIVE CONTROL PASS" if final_succ >= 0.8 else "M7 WEAK (<0.80)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
