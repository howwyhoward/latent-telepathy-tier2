"""Render a top-down still of the extruded chokepoint scene.

Scene correspondence evidence for Figure 1: the same builder the RL env and
the occlusion gate use (chokepoint/scene.py), photographed from above so a
reader can match the 3D world against the Tier 1 grid it was extruded from.

This is deliberately NOT an agent viewpoint — nothing in the scene sees this
frame. It establishes that the floor plans agree; the occlusion claim is
carried by the onboard cameras in spike/verify_occlusion.py.

Camera convention copied from spike/record_video.py's cinematic overhead:
quaternion (0,1,0,0) maps image right -> world +X (east) and image up ->
world +Y (north), matching how the Tier 1 grid is drawn.

    python spike/render_overhead.py --seed 2 --resolution 1024
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--cell", type=float, default=0.5)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--margin", type=float, default=1.18,
                    help="fraction of the arena the frame must cover; >1 keeps "
                         "the outer wall ring in shot")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.geometry import chokepoint_grid  # noqa: E402
from chokepoint.scene import build_scene_cfg  # noqa: E402

OUT_DIR = Path(__file__).parent / "out" / "hires"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    import imageio.v3 as iio

    grid = chokepoint_grid(args.seed)
    extent = grid.shape[0] * args.cell
    # f=12mm on the 20.955mm aperture Isaac assumes -> ~82.3 deg horizontal FOV
    half_fov = math.atan(20.955 / (2 * 12.0))
    height = (extent * args.margin / 2) / math.tan(half_fov)

    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60, device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode="Off"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = build_scene_cfg(
        seed=args.seed,
        cell=args.cell,
        resolution=64,
        num_envs=1,
        probe_cameras=False,
        hazards_collidable=True,
        dynamic_agents=False,
    )
    scene_cfg.cam_overhead = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, height), rot=(0.0, 1.0, 0.0, 0.0), convention="ros",
        ),
        update_period=0,
        height=args.resolution,
        width=args.resolution,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 40.0)),
    )

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    for _ in range(20):  # let the renderer settle (denoiser / exposure warmup)
        sim.step()
        scene.update(sim.get_physics_dt())

    cam: TiledCamera = scene["cam_overhead"]
    rgb = cam.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
    out = Path(args.out) if args.out else OUT_DIR / f"overhead_s{args.seed}.png"
    iio.imwrite(out, rgb)
    print(f"[overhead] seed={args.seed} extent={extent:.1f}m cam_h={height:.2f}m -> {out}",
          flush=True)

    # Kit's shutdown can hang for minutes on this box; the frame is already on
    # disk, so leave immediately rather than wait on it.
    os._exit(0)


if __name__ == "__main__":
    main()
