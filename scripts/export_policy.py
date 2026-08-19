"""Export the gated route-conditioned executor as a self-contained TorchScript
module for hardware deployment (Phase 4 plumbing test).

The artifact needs no repo code, no Isaac, no GPU:

    policy = torch.jit.load("policy_deploy.pt")
    action = policy(rgb, route)   # rgb (1,3,64,64) float32 RGB in [0,1]
                                  # route (1,2) one-hot: [1,0]=top, [0,1]=bottom
                                  # -> (1,3) mean action in [-1,1]

Scaling to hardware units is the caller's job and is recorded in the manifest:
vx = a[0]*0.5 m/s, vy = a[1]*0.5 m/s, wz = a[2]*1.5 rad/s, at 10 Hz.

Usage:
    python scripts/export_policy.py \
        --executor runs/route_obey_v6/cont.pt \
        --jepa_ckpt checkpoints/jepa_pixels.pt \
        --out export/
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chokepoint.jepa import PixelEncoder            # noqa: E402
from chokepoint.receiver import AttentionReceiver   # noqa: E402
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.geometry import CAM_H, CAM_FOCAL_MM  # noqa: E402

# Isaac camera aperture is 20.955 mm horizontal; the sim renders a square
# crop, so effective HFOV follows directly from the focal length.
HFOV_DEG = round(2.0 * math.degrees(math.atan(20.955 / (2.0 * CAM_FOCAL_MM))), 2)

ROUTE_DIM = 2  # matches rl/train_race_route.py and spike/eval_race_head.py


class DeployPolicy(nn.Module):
    """Deterministic (mean-action) wrapper around the frozen executor.

    Bakes in the empty message bus (broadcast_dim=0 -> pooled context is zero),
    so the traced graph is exactly: pixels -> frozen JEPA encoder -> receiver
    features + route bit -> actor mean, clamped to [-1, 1].
    """

    def __init__(self, executor: AttentionReceiver):
        super().__init__()
        self.executor = executor

    def forward(self, rgb: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        b = rgb.shape[0]
        empty_msg = torch.zeros(b, 0, 1, device=rgb.device)
        empty_mask = torch.zeros(b, 0, device=rgb.device)
        h = self.executor.features(rgb, empty_msg, empty_mask, route)
        return self.executor.actor(h).clamp(-1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--executor", default="runs/route_obey_v6/cont.pt")
    p.add_argument("--jepa_ckpt", default="checkpoints/jepa_pixels.pt")
    p.add_argument("--out", default="export")
    p.add_argument("--name", default="policy_deploy.pt",
                   help="output artifact filename")
    p.add_argument("--fixture_dataset",
                   default="/data/howard/isaac/datasets/chokepoint_v3_realcam20.npz",
                   help="npz of sim frames; the most route-sensitive one is "
                        "shipped as fixture_frame.npy for the robot self-test "
                        "(synthetic frames saturate the actor identically on "
                        "both routes). Pass '' to skip.")
    args = p.parse_args()

    device = torch.device("cpu")  # Jetson-friendly; the net is tiny

    ck = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ck["config"]["latent_dim"]).eval()
    encoder.load_state_dict(ck["encoder"])

    executor = AttentionReceiver(
        encoder, broadcast_dim=0, latent_dim=LATENT_DIM, route_dim=ROUTE_DIM
    )
    exec_ck = torch.load(args.executor, map_location=device)
    executor.load_state_dict(exec_ck["policy"])
    executor.eval()
    log_std = executor.log_std.detach().tolist()

    wrapper = DeployPolicy(executor).eval()

    rgb = torch.rand(1, 3, 64, 64)
    route = torch.tensor([[1.0, 0.0]])
    with torch.no_grad():
        eager = wrapper(rgb, route)
        traced = torch.jit.trace(wrapper, (rgb, route))
        scripted = traced(rgb, route)
    assert torch.allclose(eager, scripted, atol=1e-6), "trace mismatch"

    # Sanity probes: mid-gray frame (closest to sim wall colour) on both routes,
    # plus a seeded pseudo-random frame -- flat frames can saturate the actor
    # identically on both routes, which would make the fixture blind to a
    # miswired route input.
    gray = torch.full((1, 3, 64, 64), 0.5)
    seeded = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a_top = traced(gray, torch.tensor([[1.0, 0.0]]))[0].tolist()
        a_bot = traced(gray, torch.tensor([[0.0, 1.0]]))[0].tolist()
        s_top = traced(seeded, torch.tensor([[1.0, 0.0]]))[0].tolist()
        s_bot = traced(seeded, torch.tensor([[0.0, 1.0]]))[0].tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ts_path = out / args.name
    traced.save(str(ts_path))
    sha256 = hashlib.sha256(ts_path.read_bytes()).hexdigest()

    fixture = None
    if args.fixture_dataset:
        import numpy as np
        data = np.load(args.fixture_dataset)
        idx = np.linspace(0, len(data["rgb"]) - 1, 512).astype(int)
        frames = torch.from_numpy(data["rgb"][idx]).float().permute(0, 3, 1, 2) / 255.0
        top = torch.tensor([[1.0, 0.0]]).expand(len(idx), 2)
        bot = torch.tensor([[0.0, 1.0]]).expand(len(idx), 2)
        with torch.no_grad():
            diff = (traced(frames, top) - traced(frames, bot)).norm(dim=1)
        best = int(diff.argmax())
        frame = frames[best : best + 1]
        with torch.no_grad():
            f_top = traced(frame, torch.tensor([[1.0, 0.0]]))[0].tolist()
            f_bot = traced(frame, torch.tensor([[0.0, 1.0]]))[0].tolist()
        np.save(out / "fixture_frame.npy", frame.numpy())
        fixture = {
            "note": "real sim frame, shape (1,3,64,64) float32 in [0,1]; load "
                    "with np.load and feed directly. Outputs must match to 1e-3 "
                    "and MUST differ between routes (verifies route wiring).",
            "source": f"{args.fixture_dataset} frame {int(idx[best])}",
            "route_top": f_top,
            "route_bottom": f_bot,
            "route_l2_gap": float(diff[best]),
        }

    manifest = {
        "artifact": f"{args.name} (TorchScript, CPU, deterministic mean action)",
        "sha256": sha256,
        "source_executor": args.executor,
        "source_jepa": args.jepa_ckpt,
        "input_rgb": "shape (1,3,64,64), float32, RGB channel order, values in [0,1] (frame.astype(float32)/255)",
        "input_route": "shape (1,2), one-hot float32: [1,0]=top corridor, [0,1]=bottom corridor. REQUIRED.",
        "output": "shape (1,3), mean action clamped to [-1,1]: [vx_norm, vy_norm, wz_norm]",
        "action_scaling": {"vx": "a[0] * 0.5 m/s", "vy": "a[1] * 0.5 m/s", "wz": "a[2] * 1.5 rad/s"},
        "control_rate_hz": 10,
        "trained_log_std": log_std,
        "note_sampling": "Do NOT sample; use the output directly (it is already the mean).",
        "sim_camera": {
            "resolution": "64x64",
            "hfov_deg": HFOV_DEG,
            "height_m": CAM_H,
            "pitch_deg_down": 2.1,
            "renderer": "pinhole, RGB",
        },
        "sanity_gray_frame": {"route_top": a_top, "route_bottom": a_bot},
        "sanity_seeded_frame": {
            "note": "frame = torch.rand(1,3,64,64, generator=torch.Generator().manual_seed(0))",
            "route_top": s_top,
            "route_bottom": s_bot,
        },
        "fixture_frame": fixture,
        "n_actions": N_ACTIONS,
    }
    manifest_path = out / (Path(args.name).stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Reload from disk to prove the artifact is standalone.
    reloaded = torch.jit.load(str(ts_path))
    with torch.no_grad():
        assert torch.allclose(reloaded(rgb, route), eager, atol=1e-6)

    print(f"wrote {ts_path} ({ts_path.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {manifest_path}")
    print(f"sha256 {sha256}")
    print(f"trained log_std = {log_std}")
    print(f"gray-frame action    top={a_top}  bottom={a_bot}")
    print(f"seeded-frame action  top={s_top}  bottom={s_bot}")
    if fixture:
        print(f"fixture frame ({fixture['source']}), route L2 gap {fixture['route_l2_gap']:.3f}")
        print(f"  top={fixture['route_top']}\n  bottom={fixture['route_bottom']}")


if __name__ == "__main__":
    main()
