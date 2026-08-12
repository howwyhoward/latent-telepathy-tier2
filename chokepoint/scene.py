"""Single source of truth for the extruded chokepoint scene.

Both instruments import this module:
  - spike/verify_occlusion.py  (the occlusion gate)
  - chokepoint/env.py          (the RL environment)

so the geometry the gate certifies is, by construction, the geometry the
policy trains in. The grid comes from Tier 1's tested
`generate_chokepoint_map()` with ONE Tier 2 amendment — the inter-corridor
rung is sealed (see geometry.remove_rung) — then extruded to 3D.

Occlusion note: Tier 1's FOV radius has no 3D equivalent, so each corridor
carries two staggered light baffles (north-attached at BAFFLE_COLS[0],
south-attached at BAFFLE_COLS[1], each blocking BAFFLE_L of the 1.0 m
corridor width). The overlap kills every straight ray from the corridor
mouth to the hazard slab; a 0.4 m S-gap remains passable by the 0.24 m
robot. Verified empirically by the gate — rerun it after ANY edit here.
"""

import sys
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg

from .constants import AGENT_NAMES  # noqa: F401  (re-exported)
from .geometry import (  # noqa: F401  (re-exported)
    BAFFLE_COLS,
    BAFFLE_L,
    BAFFLE_T,
    CAM_H,
    HAZARD_H,
    ROBOT_SIZE,
    WALL_H,
    ChokepointGeometry,
    chokepoint_grid,
    compute_geometry,
    grid_to_world,
    remove_rung,
    wall_runs,
)

# Tier 1 is the source of truth for the map
sys.path.insert(0, str(Path.home() / "latent-telepathy"))
from envs.constants import HAZARD  # noqa: E402
from envs.map_generator import generate_chokepoint_map  # noqa: E402


def build_scene_cfg(
    seed: int = 2,
    cell: float = 0.5,
    resolution: int = 64,
    num_envs: int = 1,
    env_spacing: float | None = None,
    probe_cameras: bool = False,
    hazards_collidable: bool = False,
    dynamic_agents: bool = False,
    baffles: bool = True,
) -> InteractiveSceneCfg:
    """Extrude the Tier 1 chokepoint grid into an InteractiveSceneCfg.

    probe_cameras     : add static choice-point cameras (occlusion gate only).
    hazards_collidable: give slabs colliders (gate uses True for conservative
                        visibility; env uses False — hazards are passable).
    dynamic_agents    : robots as dynamic velocity-driven bodies (env) vs
                        kinematic set-pieces (gate renders, nothing moves).
    baffles           : False reproduces the pre-baffle straight corridor that
                        leaked the slab to the choice point. Diagnostic only —
                        the gate and the env both require True, and the gate
                        FAILS without them, which is the measurement.
    """
    map_spec = generate_chokepoint_map(np.random.default_rng(seed))
    # the rung is sealed in Tier 2 — see geometry.remove_rung for the why
    grid = remove_rung(map_spec.grid)
    size = grid.shape[0]
    mid = size // 2

    cfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing if env_spacing is not None else float(size) * cell + 2.0,
    )
    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cfg.light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )

    kinematic = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    collision = sim_utils.CollisionPropertiesCfg()

    # walls (merged runs)
    for i, (r, c0, c1) in enumerate(wall_runs(grid)):
        x0, y = grid_to_world(r, c0, size, cell)
        x1, _ = grid_to_world(r, c1, size, cell)
        setattr(
            cfg,
            f"wall_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall{i}",
                spawn=sim_utils.CuboidCfg(
                    size=((c1 - c0 + 1) * cell, cell, WALL_H),
                    rigid_props=kinematic,
                    collision_props=collision,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                    semantic_tags=[("class", "wall")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=((x0 + x1) / 2, y, WALL_H / 2)),
            ),
        )

    # occlusion baffles (see module docstring)
    corridors = [(mid - 3, mid - 2), (mid + 2, mid + 3)]
    baffle_i = 0
    for r0, r1 in corridors if baffles else []:
        _, y_north_c = grid_to_world(r0, 0, size, cell)
        _, y_south_c = grid_to_world(r1, 0, size, cell)
        y_north_edge = y_north_c + cell / 2
        y_south_edge = y_south_c - cell / 2
        for col, attach in ((BAFFLE_COLS[0], "north"), (BAFFLE_COLS[1], "south")):
            x, _ = grid_to_world(0, col, size, cell)
            y = (y_north_edge - BAFFLE_L / 2) if attach == "north" else (y_south_edge + BAFFLE_L / 2)
            setattr(
                cfg,
                f"baffle_{baffle_i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Baffle{baffle_i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(BAFFLE_T, BAFFLE_L, WALL_H),
                        rigid_props=kinematic,
                        collision_props=collision,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                        semantic_tags=[("class", "wall")],
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, WALL_H / 2)),
                ),
            )
            baffle_i += 1

    # hazard slabs. Env: no collider (passable, penalized by region check) and
    # NOT kinematic, so reset-time pose writes are the supported PhysX path.
    hazard_rigid = kinematic if hazards_collidable else sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True, kinematic_enabled=False
    )
    for i, (r, c) in enumerate(sorted(map(tuple, np.argwhere(grid == HAZARD)))):
        x, y = grid_to_world(r, c, size, cell)
        setattr(
            cfg,
            f"hazard_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Hazard{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(cell, cell, HAZARD_H),
                    rigid_props=hazard_rigid,
                    mass_props=None if hazards_collidable else sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=collision if hazards_collidable else None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                    semantic_tags=[("class", "hazard")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, HAZARD_H / 2)),
            ),
        )

    # goal pads
    for i, (r, c) in enumerate(map_spec.goals):
        x, y = grid_to_world(r, c, size, cell)
        setattr(
            cfg,
            f"goal_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Goal{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(cell, cell, 0.02),
                    rigid_props=kinematic,
                    collision_props=None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
                    semantic_tags=[("class", "goal")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.01)),
            ),
        )

    # robots + onboard cameras
    robot_rigid = (
        sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_linear_velocity=2.0,
            max_angular_velocity=6.0,
            linear_damping=2.0,
            angular_damping=2.0,
        )
        if dynamic_agents
        else kinematic
    )
    facing = {"navigator": (1.0, 0.0, 0.0, 0.0), "scout": (0.0, 0.0, 0.0, 1.0)}
    for name, (r, c) in zip(AGENT_NAMES, map_spec.agent_starts):
        x, y = grid_to_world(r, c, size, cell)
        setattr(
            cfg,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
                spawn=sim_utils.CuboidCfg(
                    size=ROBOT_SIZE,
                    rigid_props=robot_rigid,
                    mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
                    collision_props=collision,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
                    semantic_tags=[("class", "agent")],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x, y, ROBOT_SIZE[2] / 2), rot=facing[name]
                ),
            ),
        )
        setattr(
            cfg,
            f"cam_{name}",
            TiledCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}/cam",
                # ros convention: +Z is the optical axis (view direction)
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.16, 0.0, CAM_H), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
                ),
                update_period=0,
                height=resolution,
                width=resolution,
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 30.0)),
                colorize_semantic_segmentation=False,
            ),
        )

    # static probe cameras at the corridor-choice points (occlusion gate only)
    if probe_cameras:
        mouths = {
            "top": ((mid - 3 + mid - 2) / 2, 4.0),
            "bottom": ((mid + 2 + mid + 3) / 2, 4.0),
        }
        for mouth, (r, c) in mouths.items():
            x, y = grid_to_world(r, c, size, cell)
            setattr(
                cfg,
                f"cam_probe_{mouth}",
                TiledCameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Probe{mouth.capitalize()}Cam",
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(x, y, CAM_H), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
                    ),
                    update_period=0,
                    height=resolution,
                    width=resolution,
                    data_types=["rgb", "semantic_segmentation"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=12.0, clipping_range=(0.05, 30.0)
                    ),
                    colorize_semantic_segmentation=False,
                ),
            )
    return cfg
