"""Phase 0 feasibility spike: throughput of camera-based envs on wulab1.

Measures env-steps/sec with tiled cameras producing RGB + semantic
segmentation at encoder resolution, across several num_envs settings.
This number decides whether the full RL-on-pixels race is tractable or
whether we fall back to gridworld-policy + pixel-encoder validation.

Also dumps one RGB and one segmentation frame per run to spike/out/ so we
can visually confirm the semantic labels are wired correctly (the same
check that will later verify chokepoint occlusion).

Run (after `source setup/env.sh`):

    python spike/spike_fps_benchmark.py --num_envs 16
    python spike/spike_fps_benchmark.py --num_envs 1 --resolution 96
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--resolution", type=int, default=64, help="camera H=W in pixels")
parser.add_argument("--steps", type=int, default=500, help="benchmark steps after warmup")
parser.add_argument("--warmup", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def design_scene_cfg(resolution: int) -> InteractiveSceneCfg:
    """Build the scene config programmatically (clearer than @configclass for a spike)."""

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=8.0)

    # ground
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # two wall boxes forming a crude corridor; semantic class "wall"
    wall_spawn = sim_utils.CuboidCfg(
        size=(2.0, 0.2, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        semantic_tags=[("class", "wall")],
    )
    scene_cfg.wall_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallLeft",
        spawn=wall_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.25)),
    )
    scene_cfg.wall_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallRight",
        spawn=wall_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.25)),
    )

    # hazard slab, visually distinct, semantic class "hazard"
    scene_cfg.hazard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hazard",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            semantic_tags=[("class", "hazard")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.03)),
    )

    # two robot stand-ins: dynamic cubes we push around (dynamics don't matter
    # for the throughput question, only rendering does)
    robot_spawn = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
        semantic_tags=[("class", "agent")],
    )
    scene_cfg.robot_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RobotA",
        spawn=robot_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, 0.5, 0.1)),
    )
    scene_cfg.robot_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RobotB",
        spawn=robot_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, -0.5, 0.1)),
    )

    # one tiled camera per robot: RGB + semantic segmentation at encoder res
    cam_common = dict(
        update_period=0,
        height=resolution,
        width=resolution,
        data_types=["rgb", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, clipping_range=(0.05, 20.0)
        ),
        colorize_semantic_segmentation=False,
    )
    scene_cfg.cam_a = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotA/cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.15), convention="world"),
        **cam_common,
    )
    scene_cfg.cam_b = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotB/cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.15), convention="world"),
        **cam_common,
    )

    # light
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )
    return scene_cfg


def main():
    # DLSS (the default) is an AI upscaler — the encoder must see raw ray-traced
    # pixels, and 64x64 is below DLSS's minimum input resolution anyway.
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode="Off"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = design_scene_cfg(args.resolution)
    scene_cfg.num_envs = args.num_envs
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print(f"[spike] scene ready: num_envs={args.num_envs}, res={args.resolution}")

    cam_a: TiledCamera = scene["cam_a"]

    # warmup (shader compile, first renders)
    for _ in range(args.warmup):
        sim.step()
        scene.update(sim.get_physics_dt())

    # save one frame pair for visual verification
    rgb = cam_a.data.output["rgb"][0].cpu().numpy()
    seg = cam_a.data.output["semantic_segmentation"][0].cpu().numpy()
    np.save(OUT_DIR / f"rgb_e{args.num_envs}_r{args.resolution}.npy", rgb)
    np.save(OUT_DIR / f"seg_e{args.num_envs}_r{args.resolution}.npy", seg)
    try:
        import imageio.v3 as iio

        iio.imwrite(OUT_DIR / f"rgb_e{args.num_envs}_r{args.resolution}.png", rgb.astype(np.uint8))
        seg_vis = (seg.squeeze().astype(np.float32) / max(seg.max(), 1) * 255).astype(np.uint8)
        iio.imwrite(OUT_DIR / f"seg_e{args.num_envs}_r{args.resolution}.png", seg_vis)
    except ImportError:
        print("[spike] imageio not installed; saved .npy only")
    print("[spike] seg id mapping:", cam_a.data.info.get("semantic_segmentation", "n/a"))

    # timed loop
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        sim.step()
        scene.update(sim.get_physics_dt())
    torch.cuda.synchronize()
    dt = time.time() - t0

    sim_steps_per_s = args.steps / dt
    env_steps_per_s = sim_steps_per_s * args.num_envs
    print(
        f"[spike] RESULT num_envs={args.num_envs} res={args.resolution} "
        f"sim_steps/s={sim_steps_per_s:.1f} env_steps/s={env_steps_per_s:.1f} "
        f"({env_steps_per_s * 86400 / 1e6:.2f}M env-steps/GPU-day)"
    )

    # Kit's extension teardown deadlocks in headless mode; results are printed,
    # so skip graceful close and let the OS reclaim the GPU.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
