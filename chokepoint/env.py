"""Chokepoint hazard race as an Isaac Lab DirectMARLEnv.

Mapping to the Tier 1 stack:

  PettingZoo ParallelEnv          -> DirectMARLEnv (dict obs/actions per agent)
  grid actions (UP/DOWN/L/R/STAY) -> cmd_vel Box(3,): (vx, vy, wz), body frame,
                                     matching /robomaster_N/cmd_vel on the testbed
  one-hot egocentric patch        -> 64x64 RGB from the onboard TiledCamera
  map regenerated per episode     -> hazard slabs teleported to a coin-flip
                                     corridor per env at reset
  hazard = passable, penalized    -> slabs have no collider; a geometric AABB
                                     check applies the per-step penalty

The message bus stays OUTSIDE the env, exactly like Tier 1: this class returns
per-agent observations; senders/receiver wrap it at training time.

Reward is a skeleton mirroring Tier 1's shape (potential-based progress +
hazard penalty + team success bonus); coefficients live in the cfg so the
race harness can sweep them.
"""

import math

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .geometry import compute_distance_field, sample_free_positions
from .scene import AGENT_NAMES, HAZARD_H, ROBOT_SIZE, build_scene_cfg, compute_geometry

# Tier 1 is on sys.path courtesy of the scene import above.
from envs.map_generator import generate_chokepoint_map  # noqa: E402


@configclass
class ChokepointEnvCfg(DirectMARLEnvCfg):
    # simulation: physics at 60 Hz, control (and rendering) at 10 Hz
    decimation = 6
    # The geodesic start->goal path is ~11 m at v_max 0.5 m/s, so 30 s was
    # borderline even for an optimal policy; give slack for learning.
    episode_length_s = 60.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        render_interval=6,
        render=sim_utils.RenderCfg(antialiasing_mode="Off"),
    )

    # agents
    possible_agents = list(AGENT_NAMES)
    action_spaces = {name: spaces.Box(-1.0, 1.0, shape=(3,)) for name in AGENT_NAMES}
    observation_spaces = {
        name: spaces.Box(0.0, 1.0, shape=(64, 64, 3)) for name in AGENT_NAMES
    }
    state_space = 0  # no centralized state (critic input assembled by the trainer)

    # scene (canonical map seed 2; slab side is re-randomized per reset anyway)
    map_seed = 2
    cell = 0.5
    resolution = 64
    scene: InteractiveSceneCfg = None  # built in __post_init__

    # robot command limits (cmd_vel scaling)
    v_max = 0.5      # m/s
    w_max = 1.5      # rad/s

    # reward coefficients (Tier 1-shaped skeleton)
    rew_progress = 1.0        # per meter of approach toward own goal
    # Tier 1 economics: crossing the slab cost ~20% of the goal reward
    # (2 grid steps x -0.1 vs +1.0). Here crossing takes ~40 control steps
    # (2 m at 10 Hz), so -0.05/step keeps the ratio (2 vs +10). At -0.5 the
    # trained policy rationally refused to cross (0% success, slab-side split
    # 1.00/0.00) — the hazard must sting, not forbid.
    rew_hazard = -0.05        # per control step inside the hazard slab
    rew_time = -0.01          # per control step
    rew_success = 10.0        # team bonus, both robots at goals
    goal_radius = 0.3         # m

    # Which robots must reach their goals to terminate. None = the full team.
    # The M7 positive control sets ["navigator"] and parks the scout.
    success_agents: list | None = None

    # Spawn robots at uniform free poses with random yaw instead of the Tier 1
    # start poses. Data collection only (view diversity for the encoder);
    # training/eval keep the canonical starts.
    randomize_spawns = False

    def __post_init__(self):
        self.scene = build_scene_cfg(
            seed=self.map_seed,
            cell=self.cell,
            resolution=self.resolution,
            num_envs=self.scene.num_envs if self.scene else 64,
            dynamic_agents=True,
            hazards_collidable=False,
            probe_cameras=False,
        )


class ChokepointEnv(DirectMARLEnv):
    cfg: ChokepointEnvCfg

    def __init__(self, cfg: ChokepointEnvCfg, render_mode: str | None = None, **kwargs):
        self._geo = compute_geometry(seed=cfg.map_seed, cell=cfg.cell)
        super().__init__(cfg, render_mode, **kwargs)

        n = self.num_envs
        dev = self.device
        # per-env slab side: True = TOP corridor holds the hazard
        self._slab_top = torch.zeros(n, dtype=torch.bool, device=dev)
        # velocity commands cached between control steps (dict of [n, 3])
        self._cmd = {a: torch.zeros(n, 3, device=dev) for a in self.cfg.possible_agents}
        # potential-based shaping needs last distance-to-goal
        self._prev_dist = {a: torch.zeros(n, device=dev) for a in self.cfg.possible_agents}

        self._goal_pos = {
            a: torch.tensor(self._geo.goals[a], device=dev).repeat(n, 1)
            for a in self.cfg.possible_agents
        }
        self._aabb_top = torch.tensor(self._geo.hazard_aabb_top, device=dev)
        self._aabb_bot = torch.tensor(self._geo.hazard_aabb_bottom, device=dev)

        # Geodesic distance-to-goal fields for reward shaping (see geometry.py:
        # Euclidean shaping pins the robot into the second baffle's corner).
        grid = generate_chokepoint_map(np.random.default_rng(cfg.map_seed)).grid
        self._grid = grid
        self._spawn_rng = np.random.default_rng(cfg.map_seed + 12345)
        self._dfield = {}
        for a in self.cfg.possible_agents:
            f, origin, res = compute_distance_field(grid, cfg.cell, self._geo.goals[a])
            self._dfield[a] = torch.tensor(f, device=dev)
        self._dfield_origin = origin
        self._dfield_res = res

    # ------------------------------------------------------------------ setup

    def _setup_scene(self):
        # assets are declared in the scene cfg; nothing extra to spawn here
        pass

    # ------------------------------------------------------------- action path

    def _pre_physics_step(self, actions):
        # scale normalized actions to physical commands once per control step
        for a in self.cfg.possible_agents:
            act = actions[a].clamp(-1.0, 1.0)
            self._cmd[a][:, 0] = act[:, 0] * self.cfg.v_max
            self._cmd[a][:, 1] = act[:, 1] * self.cfg.v_max
            self._cmd[a][:, 2] = act[:, 2] * self.cfg.w_max

    def _apply_action(self):
        # called every PHYSICS step (decimation x per control step): rotate the
        # body-frame command by current yaw and write root velocities to PhysX.
        for a in self.cfg.possible_agents:
            robot = self.scene[a]
            quat = robot.data.root_quat_w  # (w, x, y, z)
            yaw = torch.atan2(
                2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
            c, s = torch.cos(yaw), torch.sin(yaw)
            vel = torch.zeros(self.num_envs, 6, device=self.device)
            vel[:, 0] = c * self._cmd[a][:, 0] - s * self._cmd[a][:, 1]
            vel[:, 1] = s * self._cmd[a][:, 0] + c * self._cmd[a][:, 1]
            vel[:, 5] = self._cmd[a][:, 2]
            robot.write_root_velocity_to_sim(vel)

    # ------------------------------------------------------------ observations

    def _get_observations(self):
        obs = {}
        for a in self.cfg.possible_agents:
            rgb = self.scene[f"cam_{a}"].data.output["rgb"]
            obs[a] = rgb.float() / 255.0
        return obs

    def _get_states(self):
        return None

    # ----------------------------------------------------------------- rewards

    def _local_pos(self, agent: str) -> torch.Tensor:
        """Robot xy relative to its env origin (the frame all geometry lives in)."""
        return self.scene[agent].data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _in_hazard(self, agent: str) -> torch.Tensor:
        p = self._local_pos(agent)
        box = torch.where(self._slab_top.unsqueeze(1), self._aabb_top, self._aabb_bot)
        return (
            (p[:, 0] >= box[:, 0]) & (p[:, 0] <= box[:, 1])
            & (p[:, 1] >= box[:, 2]) & (p[:, 1] <= box[:, 3])
        )

    def _dist_to_goal(self, agent: str) -> torch.Tensor:
        """Euclidean distance — success criterion only (well-defined at the goal)."""
        return torch.norm(self._local_pos(agent) - self._goal_pos[agent], dim=1)

    def _shaping_dist(self, agent: str) -> torch.Tensor:
        """Geodesic distance to goal, bilinearly sampled from the Dijkstra field."""
        f = self._dfield[agent]
        n = f.shape[0]
        idx = (self._local_pos(agent) - self._dfield_origin) / self._dfield_res
        idx = idx.clamp(0.0, n - 1.001)
        i0 = idx.floor().long()
        t = idx - i0.float()
        x0, y0 = i0[:, 0], i0[:, 1]
        tx, ty = t[:, 0], t[:, 1]
        top = f[x0, y0] * (1 - tx) + f[x0 + 1, y0] * tx
        bot = f[x0, y0 + 1] * (1 - tx) + f[x0 + 1, y0 + 1] * tx
        return top * (1 - ty) + bot * ty

    def _get_rewards(self):
        team_success = self._team_success()
        rewards = {}
        for a in self.cfg.possible_agents:
            dist = self._shaping_dist(a)
            progress = self._prev_dist[a] - dist
            self._prev_dist[a] = dist
            rewards[a] = (
                self.cfg.rew_progress * progress
                + self.cfg.rew_hazard * self._in_hazard(a).float()
                + self.cfg.rew_time
                + self.cfg.rew_success * team_success.float()
            )
        return rewards

    # ------------------------------------------------------------------- dones

    def _team_success(self) -> torch.Tensor:
        who = self.cfg.success_agents or self.cfg.possible_agents
        ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for a in who:
            ok &= self._dist_to_goal(a) < self.cfg.goal_radius
        return ok

    def _get_dones(self):
        success = self._team_success()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {a: success for a in self.cfg.possible_agents}
        time_outs = {a: time_out for a in self.cfg.possible_agents}
        return terminated, time_outs

    # ------------------------------------------------------------------- reset

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device
        origins = self.scene.env_origins[env_ids]

        # robots back to Tier 1 start poses (or random free poses for data
        # collection), zero velocity
        for a in self.cfg.possible_agents:
            pose = torch.zeros(n, 7, device=dev)
            if self.cfg.randomize_spawns:
                xy = sample_free_positions(self._grid, self.cfg.cell, n, self._spawn_rng)
                yaw_r = self._spawn_rng.uniform(-math.pi, math.pi, size=n)
                pose[:, 0] = origins[:, 0] + torch.tensor(xy[:, 0], dtype=torch.float32, device=dev)
                pose[:, 1] = origins[:, 1] + torch.tensor(xy[:, 1], dtype=torch.float32, device=dev)
                pose[:, 3] = torch.tensor(np.cos(yaw_r / 2), dtype=torch.float32, device=dev)
                pose[:, 6] = torch.tensor(np.sin(yaw_r / 2), dtype=torch.float32, device=dev)
            else:
                x, y, yaw = self._geo.starts[a]
                pose[:, 0] = origins[:, 0] + x
                pose[:, 1] = origins[:, 1] + y
                pose[:, 3] = math.cos(yaw / 2)
                pose[:, 6] = math.sin(yaw / 2)
            pose[:, 2] = ROBOT_SIZE[2] / 2
            robot = self.scene[a]
            robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # coin-flip the hazard corridor per env; teleport every slab cell
        side_top = torch.rand(n, device=dev) < 0.5
        self._slab_top[env_ids] = side_top
        for i, (x, y_top, y_bot) in enumerate(self._geo.slab_cells):
            slab = self.scene[f"hazard_{i}"]
            pose = torch.zeros(n, 7, device=dev)
            pose[:, 0] = origins[:, 0] + x
            pose[:, 1] = origins[:, 1] + torch.where(
                side_top,
                torch.full((n,), y_top, device=dev),
                torch.full((n,), y_bot, device=dev),
            )
            pose[:, 2] = HAZARD_H / 2
            pose[:, 3] = 1.0
            slab.write_root_pose_to_sim(pose, env_ids=env_ids)
            slab.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # re-seed the shaping potential from the fresh poses
        for a in self.cfg.possible_agents:
            self._prev_dist[a][env_ids] = self._shaping_dist(a)[env_ids]
