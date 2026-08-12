"""Phase 1 gate: empirical occlusion check for the extruded chokepoint scene.

The scene comes from chokepoint/scene.py — the SAME builder the RL env uses —
so this gate certifies exactly the geometry the policy trains in. Rerun after
ANY scene edit.

Gates (pre-registered):
  - navigator at start        : hazard pixels == 0
  - scout at start            : hazard pixels > 0 iff slab is in its (top)
                                corridor; == 0 otherwise (absence is signal)
  - choice-point probes (both corridor mouths, facing east): hazard px == 0
    This is the strict gate: Tier 1's FOV radius has no 3D equivalent, so the
    staggered baffles in the scene are what enforce it. A straight corridor
    leaked 10 px before the baffles existed.

Run (after `source setup/env.sh`):

    python spike/verify_occlusion.py --seed 0   # slab BOTTOM
    python spike/verify_occlusion.py --seed 2   # slab TOP

--no_baffles reproduces the pre-baffle straight corridor and is EXPECTED to
fail the choice-point gate; that failure is the measurement the baffles were
designed against. --tag suffixes the frame filenames and --json_out records the
counts, so a figure can be rebuilt from committed artifacts without a rerun.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resolution", type=int, default=64, help="camera H=W (encoder res)")
parser.add_argument("--cell", type=float, default=0.5, help="meters per grid cell")
parser.add_argument("--no_baffles", action="store_true",
                    help="diagnostic: build the straight corridor that leaked")
parser.add_argument("--tag", type=str, default="",
                    help="suffix for output frame filenames")
parser.add_argument("--json_out", type=str, default=None,
                    help="write the pixel counts and gate verdicts here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import TiledCamera

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.scene import build_scene_cfg  # noqa: E402

# Tier 1 constants for reporting which corridor holds the slab
sys.path.insert(0, str(Path.home() / "latent-telepathy"))
from envs.constants import HAZARD  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def hazard_pixel_count(cam: TiledCamera) -> int:
    seg = cam.data.output["semantic_segmentation"][0].cpu().numpy().squeeze()
    id_to_labels = cam.data.info["semantic_segmentation"]["idToLabels"]
    hazard_ids = [int(k) for k, v in id_to_labels.items() if v.get("class") == "hazard"]
    return int(np.isin(seg, hazard_ids).sum()) if hazard_ids else 0


def save_frames(cam: TiledCamera, name: str):
    import imageio.v3 as iio

    rgb = cam.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
    seg = cam.data.output["semantic_segmentation"][0].cpu().numpy().squeeze()
    iio.imwrite(OUT_DIR / f"occl_{name}{args.tag}_rgb.png", rgb)
    seg_vis = (seg.astype(np.float32) / max(seg.max(), 1) * 255).astype(np.uint8)
    iio.imwrite(OUT_DIR / f"occl_{name}{args.tag}_seg.png", seg_vis)
    # the hazard mask is what the gate counts; save it so a figure can overlay
    # the counted pixels rather than re-deriving them from a normalized preview
    id_to_labels = cam.data.info["semantic_segmentation"]["idToLabels"]
    hazard_ids = [int(k) for k, v in id_to_labels.items() if v.get("class") == "hazard"]
    mask = np.isin(seg, hazard_ids) if hazard_ids else np.zeros_like(seg, bool)
    np.save(OUT_DIR / f"occl_{name}{args.tag}_hazmask.npy", mask)


def main():
    grid = generate_chokepoint_map(np.random.default_rng(args.seed)).grid
    hazard_rows = sorted({r for r, _ in map(tuple, np.argwhere(grid == HAZARD))})
    slab_side = "TOP" if hazard_rows[0] < grid.shape[0] // 2 else "BOTTOM"
    print(f"[occl] seed={args.seed}: hazard slab in {slab_side} corridor (rows {hazard_rows})")

    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode="Off"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = build_scene_cfg(
        seed=args.seed,
        cell=args.cell,
        resolution=args.resolution,
        num_envs=1,
        probe_cameras=True,
        hazards_collidable=True,   # gate keeps colliders: conservative visibility
        dynamic_agents=False,      # nothing moves during the gate
        baffles=not args.no_baffles,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # a few steps so the renderer settles (denoiser/exposure warmup)
    for _ in range(20):
        sim.step()
        scene.update(sim.get_physics_dt())

    total = args.resolution ** 2
    results = {}
    for name in ("navigator", "scout"):
        cam: TiledCamera = scene[f"cam_{name}"]
        results[name] = hazard_pixel_count(cam)
        save_frames(cam, name)
    print(f"[occl] navigator hazard pixels: {results['navigator']}/{total}")
    print(f"[occl] scout     hazard pixels: {results['scout']}/{total}")

    nav_ok = results["navigator"] == 0
    if slab_side == "TOP":
        scout_ok = results["scout"] > 0
        scout_gate = "scout>0 (slab in its corridor)"
    else:
        scout_ok = results["scout"] == 0
        scout_gate = "scout==0 (slab in other corridor)"
    print(f"[occl] gate navigator==0: {'PASS' if nav_ok else 'FAIL'}")
    print(f"[occl] gate {scout_gate}: {'PASS' if scout_ok else 'FAIL'}")

    choice = {}
    for mouth in ("top", "bottom"):
        cam: TiledCamera = scene[f"cam_probe_{mouth}"]
        choice[mouth] = hazard_pixel_count(cam)
        save_frames(cam, f"navigator_mouth_{mouth}")
        print(f"[occl] choice point, {mouth} mouth: hazard pixels {choice[mouth]}/{total}")
    choice_ok = all(v == 0 for v in choice.values())
    print(f"[occl] gate choice-points==0: {'PASS' if choice_ok else 'FAIL'}")

    overall = nav_ok and scout_ok and choice_ok
    print(f"[occl] OVERALL: {'PASS' if overall else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "seed": args.seed,
                "resolution": args.resolution,
                "baffles": not args.no_baffles,
                "slab_side": slab_side,
                "total_pixels": total,
                "hazard_pixels": {
                    "navigator": results["navigator"],
                    "scout": results["scout"],
                    "probe_top": choice["top"],
                    "probe_bottom": choice["bottom"],
                },
                "gates": {"navigator": nav_ok, "scout": scout_ok,
                          "choice_points": choice_ok, "overall": overall},
            }, f, indent=2)
        print(f"[occl] wrote {args.json_out}")

    # Kit's extension teardown deadlocks in headless mode; results are printed,
    # so skip graceful close and let the OS reclaim the GPU.
    sys.stdout.flush()
    os._exit(0 if overall else 1)


if __name__ == "__main__":
    main()
