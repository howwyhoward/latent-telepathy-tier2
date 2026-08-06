"""WP7 prep — fit the slab-side probe on SIM latents and save it to disk.

The sim-to-real gate measurement (spike/eval_real_frames.py) must run the
moment real frames arrive, without Isaac. This script does the Isaac half now:
collect scout latents over randomized slab sides, fit the logistic probe, and
persist (w, b), held-out accuracy, and the sim latent statistics the health
check compares against.

    python spike/fit_slab_probe.py --out checkpoints/slab_probe.pt
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--probe_episodes", type=int, default=512)
parser.add_argument("--out", type=str, default="checkpoints/slab_probe.pt")
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402

LEARNER, BEACON = "navigator", "scout"


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.success_agents = [LEARNER]
    env = ChokepointEnv(cfg)
    device = env.device
    E = args.num_envs

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    zero = torch.zeros(E, N_ACTIONS, device=device)
    zs, ys = [], []
    seen = 0
    while seen < args.probe_episodes:
        env.reset()
        # reset-frame staleness: cameras render during stepping
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step({LEARNER: zero, BEACON: zero})
        rgb = obs_dict[BEACON].permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            zs.append(encoder(rgb))
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "w": w.detach().cpu(),
        "b": b.detach().cpu(),
        "acc_train": acc_tr,
        "acc_heldout": acc_te,
        "n_samples": len(Z),
        # reference statistics for the real-frame latent health check
        "sim_latent_mean": Z.mean(dim=0).cpu(),
        "sim_latent_std": Z.std(dim=0).cpu(),
        "jepa_ckpt": args.jepa_ckpt,
    }, args.out)
    print(f"saved -> {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
