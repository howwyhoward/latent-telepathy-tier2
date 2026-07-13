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
                    choices=["none", "position", "z_t", "raw"])
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0,
                    help="always-in-range for the race (content, not range, is the contrast)")
parser.add_argument("--total_timesteps", type=int, default=3_000_000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=128)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.0)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.message_bus import (  # noqa: E402
    LatentBroadcast,
    MessageBus,
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
    logprobs = torch.zeros((S, E), device=device)
    rewards = torch.zeros((S, E), device=device)
    dones = torch.zeros((S, E), device=device)
    values = torch.zeros((S, E), device=device)

    next_obs, next_msg, next_mask = nav_inputs(obs_dict)
    next_done = torch.zeros(E, device=device)

    acc_ret = torch.zeros(E, device=device)
    acc_len = torch.zeros(E, device=device)
    acc_hazard = torch.zeros(E, device=device)
    ep_returns: deque = deque(maxlen=100)
    ep_lengths: deque = deque(maxlen=100)
    ep_successes: deque = deque(maxlen=100)
    ep_hazards: deque = deque(maxlen=100)
    ep_succ_slab_top: deque = deque(maxlen=100)
    ep_succ_slab_bot: deque = deque(maxlen=100)
    ep_slab = env._slab_top.clone()
    curve = []

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

        for step in range(S):
            global_step += E
            obs[step] = next_obs
            msgs[step] = next_msg
            masks[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    next_obs, next_msg, next_mask
                )
            values[step] = value.flatten()
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
                ep_slab[idx] = env._slab_top[idx]

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
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb], b_msgs[mb], b_masks[mb], b_actions[mb]
                )
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
        curve.append((global_step, float(succ), float(mean_ret), float(mean_len), float(mean_haz)))
        print(
            f"[{args.condition}] iter {iteration:3d}/{args.num_iterations}  "
            f"step {global_step:>8d}  return {mean_ret:7.3f}  "
            f"success {succ:5.2f} (top {s_top:4.2f}/bot {s_bot:4.2f})  "
            f"ep_len {mean_len:6.1f}  hazard_steps {mean_haz:5.1f}  "
            f"approx_kl {approx_kl.item():.4f}  SPS {sps}",
            flush=True,
        )
        if csv_file is not None:
            csv_file.write(
                f"{global_step},{succ},{s_top},{s_bot},{mean_len},{mean_ret},"
                f"{mean_haz},{approx_kl.item()},{sps}\n"
            )
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    final_succ = float(np.mean(ep_successes)) if ep_successes else 0.0
    final_haz = float(np.mean(ep_hazards)) if ep_hazards else float("nan")
    print(f"\n[{args.condition}/seed{args.seed}] FINAL success {final_succ:.3f}  "
          f"hazard_steps {final_haz:.2f}")

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
                },
                f,
                indent=2,
            )
        print(f"wrote run json -> {args.run_json}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
