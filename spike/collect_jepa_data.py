"""Stage-A data collection: random-policy rollouts for JEPA training.

Faithful to Tier 1's recipe (random policy, cached transitions), adapted to
pixels. Robots spawn at uniform free poses with random yaw every short
episode (view diversity — Tier 1 got this for free from map regeneration),
drive with temporally-correlated random cmd_vel, and we record per agent per
control step:

  rgb    (N, 64, 64, 3) uint8   onboard camera
  seg    (N, 64, 64)    uint8   ground-truth semantics, remapped to
                                jepa.SEG_CLASSES indices
  action (N, 3)         float32 normalized cmd_vel applied AT this frame
  valid  (N,)           bool    True if frame t+num_streams is the true
                                successor (False across episode resets)

Frames are stored step-major: streams = num_envs * n_agents, sample index
= t * streams + stream. The trainer forms (rgb_t, a_t, rgb_t+1) pairs via
the valid mask. Both agents feed one dataset — the encoder is shared.

Run:  python spike/collect_jepa_data.py --num_envs 64 --steps 1600 \
          --out /data/howard/isaac/datasets/chokepoint_v1.npz
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1600, help="control steps to record")
parser.add_argument("--episode_s", type=float, default=5.0, help="respawn interval")
parser.add_argument("--action_repeat", type=int, default=3,
                    help="hold each random cmd_vel this many control steps")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str,
                    default="/data/howard/isaac/datasets/chokepoint_v1.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time
import traceback


def _die_loudly(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _die_loudly

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import SEG_CLASSES  # noqa: E402


def seg_remap_table(cam) -> np.ndarray:
    """Sim segmentation ids -> SEG_CLASSES indices (unknown ids -> background)."""
    id_to_labels = cam.data.info["semantic_segmentation"]["idToLabels"]
    max_id = max(int(k) for k in id_to_labels)
    table = np.zeros(max_id + 1, dtype=np.uint8)  # 0 == background
    for k, v in id_to_labels.items():
        cls = v.get("class")
        if cls in SEG_CLASSES:
            table[int(k)] = SEG_CLASSES.index(cls)
    return table


def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.randomize_spawns = True
    cfg.episode_length_s = args.episode_s
    env = ChokepointEnv(cfg)
    agents = list(env.cfg.possible_agents)
    obs_dict, _ = env.reset()

    streams = args.num_envs * len(agents)
    n_total = args.steps * streams
    rgb_buf = np.empty((n_total, 64, 64, 3), dtype=np.uint8)
    seg_buf = np.empty((n_total, 64, 64), dtype=np.uint8)
    act_buf = np.empty((n_total, 3), dtype=np.float32)
    valid_buf = np.zeros(n_total, dtype=bool)

    tables = {a: None for a in agents}
    actions = {
        a: torch.zeros(args.num_envs, 3, device=env.device) for a in agents
    }

    t0 = time.time()
    for t in range(args.steps):
        if t % args.action_repeat == 0:
            for a in agents:
                actions[a] = (
                    torch.rand(args.num_envs, 3, device=env.device) * 2 - 1
                )

        # record the frame the policy acts on, plus the action taken from it
        for ai, a in enumerate(agents):
            cam = env.scene[f"cam_{a}"]
            # idToLabels grows as classes first enter view - rebuild every step
            tables[a] = seg_remap_table(cam)
            rgb = (obs_dict[a] * 255).to(torch.uint8).cpu().numpy()
            seg = cam.data.output["semantic_segmentation"].cpu().numpy().squeeze(-1)
            sl = slice(
                t * streams + ai * args.num_envs,
                t * streams + (ai + 1) * args.num_envs,
            )
            rgb_buf[sl] = rgb
            seg_buf[sl] = tables[a][np.clip(seg, 0, len(tables[a]) - 1)]
            act_buf[sl] = actions[a].cpu().numpy()

        obs_dict, _, term, tout, _ = env.step(actions)
        done = (term[agents[0]] | tout[agents[0]]).cpu().numpy()

        # frame t's successor (t+1) is real only if the episode didn't reset
        if t + 1 < args.steps:
            for ai in range(len(agents)):
                sl = slice(
                    t * streams + ai * args.num_envs,
                    t * streams + (ai + 1) * args.num_envs,
                )
                valid_buf[sl] = ~done

        if (t + 1) % 200 == 0:
            fps = (t + 1) * streams / (time.time() - t0)
            print(f"[collect] step {t + 1}/{args.steps}  ({fps:.0f} frames/s)", flush=True)

    counts = np.bincount(seg_buf.reshape(-1), minlength=len(SEG_CLASSES))
    frac = counts / counts.sum()
    print("[collect] class pixel fractions:",
          {c: f"{frac[i]:.4f}" for i, c in enumerate(SEG_CLASSES)})
    for i, c in enumerate(SEG_CLASSES):
        if c != "background":
            per_frame = (seg_buf == i).any(axis=(1, 2)).mean()
            print(f"[collect] frames with any '{c}' pixel: {per_frame:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rgb=rgb_buf,
        seg=seg_buf,
        action=act_buf,
        valid=valid_buf,
        streams=streams,
        num_envs=args.num_envs,
        agents=np.array(agents),
        seg_classes=np.array(SEG_CLASSES),
        seed=args.seed,
    )
    print(f"[collect] wrote {out} ({out.stat().st_size / 1e9:.2f} GB, "
          f"{n_total} frames, {int(valid_buf.sum())} valid transitions)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
