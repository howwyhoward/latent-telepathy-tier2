"""Post-mortem for a trained M7 agent: where does the navigator actually go?

Loads runs/m7_agent.pt, rolls out full episodes with the DETERMINISTIC policy
(mean action, no sampling), records the navigator's env-local xy every control
step, and renders all trajectories over the map geometry. Also prints the
stall point (final position) per env and its distance to goal.

Run:  python spike/diag_rollout.py --num_envs 8 --ckpt runs/m7_agent.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="runs/m7_agent.pt")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--stochastic", action="store_true", help="sample instead of mean action")
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
from chokepoint.agent import LEARNER, Agent  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.scene import BAFFLE_COLS, BAFFLE_L, BAFFLE_T, grid_to_world, wall_runs  # noqa: E402

sys.path.insert(0, str(Path.home() / "latent-telepathy"))
from envs.map_generator import generate_chokepoint_map  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device

    agent = Agent().to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()
    zero_scout = torch.zeros(args.num_envs, 3, device=device)

    obs_dict, _ = env.reset()
    traj = np.zeros((args.steps, args.num_envs, 2), dtype=np.float32)
    done_at = np.full(args.num_envs, args.steps, dtype=int)
    succeeded = np.zeros(args.num_envs, dtype=bool)

    for t in range(args.steps):
        traj[t] = env._local_pos(LEARNER).cpu().numpy()
        with torch.no_grad():
            x = obs_dict[LEARNER].permute(0, 3, 1, 2).contiguous()
            if args.stochastic:
                action, _, _, _ = agent.get_action_and_value(x)
            else:
                action = agent.actor_mean(agent.trunk(x))
        obs_dict, _, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), "scout": zero_scout}
        )
        d = (term[LEARNER] | tout[LEARNER]).cpu().numpy()
        first_done = d & (done_at == args.steps)
        done_at[first_done] = t
        succeeded |= term[LEARNER].cpu().numpy() & first_done

    goal = np.array(env._geo.goals[LEARNER])
    print(f"[diag] goal at {goal}, start at {env._geo.starts[LEARNER][:2]}")
    for i in range(args.num_envs):
        end = traj[min(done_at[i], args.steps - 1), i]
        print(
            f"[diag] env {i}: slab_top={bool(env._slab_top[i])} "
            f"end=({end[0]:+.2f},{end[1]:+.2f}) dist_to_goal={np.linalg.norm(end - goal):.2f} "
            f"{'SUCCESS at step ' + str(done_at[i]) if succeeded[i] else 'timeout'}"
        )

    # ---- plot trajectories over the geometry -------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = generate_chokepoint_map(np.random.default_rng(cfg.map_seed)).grid
    size = grid.shape[0]
    cell = cfg.cell
    mid = size // 2

    fig, ax = plt.subplots(figsize=(9, 9))
    for r, c0, c1 in wall_runs(grid):
        x0, y = grid_to_world(r, c0, size, cell)
        x1, _ = grid_to_world(r, c1, size, cell)
        ax.add_patch(
            plt.Rectangle(
                (x0 - cell / 2, y - cell / 2), (c1 - c0 + 1) * cell, cell,
                color="0.4", zorder=1,
            )
        )
    for r0, r1 in [(mid - 3, mid - 2), (mid + 2, mid + 3)]:
        _, y_n = grid_to_world(r0, 0, size, cell)
        _, y_s = grid_to_world(r1, 0, size, cell)
        for col, attach in ((BAFFLE_COLS[0], "north"), (BAFFLE_COLS[1], "south")):
            x, _ = grid_to_world(0, col, size, cell)
            yc = (y_n + cell / 2 - BAFFLE_L / 2) if attach == "north" else (y_s - cell / 2 + BAFFLE_L / 2)
            ax.add_patch(
                plt.Rectangle(
                    (x - BAFFLE_T / 2, yc - BAFFLE_L / 2), BAFFLE_T, BAFFLE_L,
                    color="0.15", zorder=2,
                )
            )
    for side, box in (("top", env._geo.hazard_aabb_top), ("bottom", env._geo.hazard_aabb_bottom)):
        ax.add_patch(
            plt.Rectangle(
                (box[0], box[2]), box[1] - box[0], box[3] - box[2],
                color="red", alpha=0.15, zorder=1, label=f"hazard {side} (candidate)",
            )
        )
    ax.plot(*goal, "g*", markersize=20, zorder=5, label="goal")
    sx, sy, _ = env._geo.starts[LEARNER]
    ax.plot(sx, sy, "bs", markersize=10, zorder=5, label="start")

    cmap = plt.cm.viridis(np.linspace(0, 1, args.num_envs))
    for i in range(args.num_envs):
        end_t = min(done_at[i] + 1, args.steps)
        ax.plot(traj[:end_t, i, 0], traj[:end_t, i, 1], "-", color=cmap[i], lw=1.2,
                alpha=0.85, zorder=3)
        ax.plot(*traj[end_t - 1, i], "o", color=cmap[i], markersize=6, zorder=4)

    ax.set_aspect("equal")
    ax.set_title(f"M7 deterministic rollouts ({args.num_envs} envs, dots = final pose)")
    ax.legend(loc="upper left", fontsize=8)
    out = OUT / "diag_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[diag] wrote {out}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
