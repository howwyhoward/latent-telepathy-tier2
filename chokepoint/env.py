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

from .geometry import (
    chokepoint_grid,
    compute_distance_field,
    corridor_rect,
    descend_field,
    route_distance_fields,
    sample_free_positions,
    sample_free_positions_band,
)
from .scene import AGENT_NAMES, HAZARD_H, ROBOT_SIZE, build_scene_cfg, compute_geometry


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
    # One-time penalty on slab ENTRY (rising edge). Per-step penalties deep
    # enough to matter (-0.15) create a ~43-step penalty valley the policy
    # refuses to cross (race v2: every condition timed out rather than cross);
    # an entry cost is sunk the moment it lands, so the gradient at the far
    # edge points forward, not back. 0.0 preserves M7/v1 economics.
    rew_hazard_entry = 0.0
    rew_time = -0.01          # per control step
    rew_success = 10.0        # team bonus, both robots at goals
    # Round-5 lever (19 Aug audit): rounds 2-4 bought obedience with success --
    # the global wrong-corridor penalty and the abort suppress goal-reaching
    # gradients everywhere, and every run peaked mid-run then declined. This
    # instead gates the success BONUS on never having committed to the wrong
    # corridor: disobedient arrivals earn nothing, obedient ones keep the full
    # unpenalized gradient. Only meaningful with route_instruction.
    rew_success_obedient_only = False
    goal_radius = 0.3         # m

    # Which robots must reach their goals to terminate. None = the full team.
    # The M7 positive control sets ["navigator"] and parks the scout.
    success_agents: list | None = None

    # Pin the hazard slab to one corridor instead of the per-episode coin
    # flip (True = top, False = bottom). Eval/visualization only — training
    # must keep the coin flip or the corridor decision becomes trivial.
    force_slab_top: bool | None = None

    # Spawn robots at uniform free poses with random yaw instead of the Tier 1
    # start poses. Data collection only (view diversity for the encoder);
    # training/eval keep the canonical starts.
    randomize_spawns = False

    # Spawn CURRICULUM (race v5): with this probability, an episode starts the
    # navigator at a random free pose instead of the canonical west-chamber
    # start (scout untouched). Gaussian cmd_vel noise never flips a committed
    # macro-route, so corridor choice is bistable: v4 runs froze into either
    # blind-crossing or refusal by early-training luck, with approx_kl -> 0
    # long before the message channel could be recruited. Random spawns make
    # BOTH corridors (and the far side of the slab) trained territory, turning
    # the west-chamber decision into a choice between two known-good routes —
    # the gradient the message needs. Race metrics must be computed on
    # canonical-start episodes only (see `curriculum_spawn` mask).
    spawn_curriculum_prob = 0.0

    # REVERSE-curriculum band: when set to (d_lo, d_hi) meters, curriculum
    # spawns are restricted to free poses whose GEODESIC distance to the
    # navigator's goal lies in the band. Uniform-over-the-map spawning (v5.5)
    # taught nothing: spawns near the goal are trivial, spawns elsewhere are
    # unreachable-in-practice, and corridor traversal never got a dense
    # signal (trunk eval: east chamber 0.73, every corridor region 0.00).
    # The trainer anneals d_hi from ~2 m to past the canonical start, walking
    # the learning frontier backwards through the corridors.
    spawn_dist_range: tuple | None = None

    # MOUTH curriculum (stage-1 v3): with this probability the navigator
    # spawns at a corridor mouth (coin-flip top/bottom), FACING EAST. Both
    # free-pose curricula failed the trunk gate: with random pose + random
    # yaw the policy never learned corridor traversal at all (banded run:
    # 0.88 near-goal -> 0.26 as the band widened; every corridor region
    # 0.00), while v4 proved this same policy class learns a fixed-pose,
    # fixed-heading route to 0.99. So stage 1 trains ONLY the poses stage 2
    # needs: canonical start + the two mouths, stationary mixture, fixed
    # headings. Mouth spawns force both corridors to be traversed and valued.
    spawn_mouth_prob = 0.0

    # ROUTE OBEDIENCE (stage 1.5). Races v1-v7 all failed the same way: the
    # policy has no representation of "which corridor am I taking", so a route
    # preference can only live in the lateral component of a 600-sample Gaussian
    # — exactly where the 1/sigma^2 log-prob gradient guarantees it loses to the
    # constant-view bias (v7: coverage of the far corridor reached 0.22 and was
    # optimized away by iteration 80). The v7 oracle did recruit the bit, but to
    # gate advance-vs-balk inside the corridor it was already taking (lying to
    # it costs 1.00 -> 0.41 success without ever changing the corridor).
    # So the route becomes an explicit instruction the policy is trained to
    # OBEY here, and learns to SELECT in stage 2. Obedience gets a dense,
    # immediate penalty for occupying the corridor it was not sent down, which
    # is what breaks the chicken-and-egg: a route input that changes nothing
    # earns no gradient, and a route head whose choice changes nothing earns
    # none either.
    route_instruction = False   # True: publish a per-episode route command
    rew_wrong_corridor = -0.25  # per step spent in the corridor NOT instructed
    route_shaping = True        # shape progress via the commanded corridor
    route_abort_wrong = False   # end the episode on entering the wrong corridor
    # Reverse curriculum along the commanded route. frac 1.0 spawns at that
    # corridor's mouth, 0.0 at the canonical start; anneal it down so the spawn
    # walks back into the chamber over the leg that has never been trained.
    spawn_route_prob = 0.0
    spawn_route_frac = 1.0
    # Uniform heading jitter (radians) on navigator spawns. Two reasons: the
    # curric run showed the residual failure is purely the initial heading
    # (same position, 45 deg yaw -> ~1.0 obedience; 0 deg -> 0.0), and a real
    # RoboMaster is never placed at exactly 0 deg anyway.
    spawn_yaw_jitter = 0.0

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
        # per-env route command: True = navigator is told to take the TOP
        # corridor. Set to the SAFE side, so obeying is never in tension with
        # the hazard and both routes are commanded equally often.
        self._route_top = torch.zeros(n, dtype=torch.bool, device=dev)
        # velocity commands cached between control steps (dict of [n, 3])
        self._cmd = {a: torch.zeros(n, 3, device=dev) for a in self.cfg.possible_agents}
        # potential-based shaping needs last distance-to-goal
        self._prev_dist = {a: torch.zeros(n, device=dev) for a in self.cfg.possible_agents}
        # hazard-entry detection (rising edge of the in-hazard flag)
        self._prev_in_hazard = {
            a: torch.zeros(n, dtype=torch.bool, device=dev)
            for a in self.cfg.possible_agents
        }
        # True where the running episode used a curriculum (random) spawn
        self.curriculum_spawn = torch.zeros(n, dtype=torch.bool, device=dev)
        # latch: navigator has entered the wrong corridor this episode
        self._committed_wrong = torch.zeros(n, dtype=torch.bool, device=dev)

        self._goal_pos = {
            a: torch.tensor(self._geo.goals[a], device=dev).repeat(n, 1)
            for a in self.cfg.possible_agents
        }
        self._aabb_top = torch.tensor(self._geo.hazard_aabb_top, device=dev)
        self._aabb_bot = torch.tensor(self._geo.hazard_aabb_bottom, device=dev)
        self._corr_top = torch.tensor(self._geo.corridor_top, device=dev)
        self._corr_bot = torch.tensor(self._geo.corridor_bottom, device=dev)

        # Geodesic distance-to-goal fields for reward shaping (see geometry.py:
        # Euclidean shaping pins the robot into the second baffle's corner).
        grid = chokepoint_grid(cfg.map_seed)
        self._grid = grid
        self._spawn_rng = np.random.default_rng(cfg.map_seed + 12345)
        self._dfield = {}
        for a in self.cfg.possible_agents:
            f, origin, res = compute_distance_field(grid, cfg.cell, self._geo.goals[a])
            self._dfield[a] = torch.tensor(f, device=dev)
        self._dfield_origin = origin
        self._dfield_res = res

        # Route-conditioned shaping (stage 1.5). Eight generations of penalties
        # taught the policy to STOP when the wrong corridor was punished, never
        # to turn: the chamber-to-top-mouth leg was never once trained, so
        # "stop" was the only response it could express. These fields make the
        # commanded route downhill from the very first step.
        self._dfield_route = None
        self._route_poses = None
        if cfg.route_instruction:
            fields, r_origin, r_res = route_distance_fields(
                grid, cfg.cell, self._geo.goals["navigator"]
            )
            assert (r_origin, r_res) == (origin, res), "route field grid mismatch"
            if cfg.route_shaping:
                self._dfield_route = {
                    k: torch.tensor(v, device=dev) for k, v in fields.items()
                }
            # Curriculum poses: the field's own descent path from the canonical
            # start to the commanded corridor, then the mouth pose that stage 1
            # trained (2.5 m west, facing east), which the policy already
            # executes reliably. A straight line between the two would graze the
            # divider corner; the descent path stays in free space.
            sx, sy, _ = self._geo.starts["navigator"]
            self._route_poses = {}
            for take_top in (True, False):
                path = descend_field(
                    fields[take_top], origin, res, (sx, sy),
                    stop_rect=corridor_rect(grid, cfg.cell, top=take_top),
                )
                mouth = np.array([[-2.5, 1.0 if take_top else -1.5, 0.0]], dtype=np.float32)
                self._route_poses[take_top] = torch.tensor(
                    np.concatenate([path, mouth]), device=dev
                )

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

    @staticmethod
    def _inside(p: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return (
            (p[:, 0] >= box[0]) & (p[:, 0] <= box[1])
            & (p[:, 1] >= box[2]) & (p[:, 1] <= box[3])
        )

    def in_corridor(self, agent: str, top: bool) -> torch.Tensor:
        """Is the robot inside the corridor proper (not the chambers)?"""
        p = self._local_pos(agent)
        return self._inside(p, self._corr_top if top else self._corr_bot)

    def in_wrong_corridor(self, agent: str) -> torch.Tensor:
        """Inside the corridor the route command did NOT ask for."""
        return torch.where(
            self._route_top,
            self.in_corridor(agent, top=False),
            self.in_corridor(agent, top=True),
        )

    def route_onehot(self) -> torch.Tensor:
        """The route command as the policy sees it: [is_top, is_bottom]."""
        top = self._route_top.float().unsqueeze(1)
        return torch.cat([top, 1.0 - top], dim=1)

    def _dist_to_goal(self, agent: str) -> torch.Tensor:
        """Euclidean distance — success criterion only (well-defined at the goal)."""
        return torch.norm(self._local_pos(agent) - self._goal_pos[agent], dim=1)

    def _shaping_dist(self, agent: str) -> torch.Tensor:
        """Geodesic distance to goal, bilinearly sampled from the Dijkstra field.

        Under a route command the navigator is shaped against the field that
        routes VIA the commanded corridor, so progress toward the right corridor
        pays from the first step instead of having to be discovered.
        """
        if self._dfield_route is not None and agent == "navigator":
            return torch.where(
                self._route_top,
                self._sample_field(self._dfield_route[True], agent),
                self._sample_field(self._dfield_route[False], agent),
            )
        return self._sample_field(self._dfield[agent], agent)

    def _sample_field(self, f: torch.Tensor, agent: str) -> torch.Tensor:
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
        if self.cfg.route_instruction:
            self._committed_wrong |= self.in_wrong_corridor("navigator")
        rewards = {}
        for a in self.cfg.possible_agents:
            dist = self._shaping_dist(a)
            # A control step covers at most v_max * dt metres, so any larger
            # jump in the field is a discontinuity, not travel: obstacle cells
            # carry a fill value, and a route field switches metric at the wrong
            # corridor's far mouth. Left unclamped those cliffs pay out tens of
            # metres in one step, which is how the x8 route field trained the
            # policy to bulldoze the hazard corridor.
            step_max = 1.5 * self.cfg.v_max * self.step_dt
            progress = (self._prev_dist[a] - dist).clamp(-step_max, step_max)
            self._prev_dist[a] = dist
            in_haz = self._in_hazard(a)
            entered = in_haz & ~self._prev_in_hazard[a]
            self._prev_in_hazard[a] = in_haz
            bonus = team_success
            if (a == "navigator" and self.cfg.route_instruction
                    and self.cfg.rew_success_obedient_only):
                bonus = team_success & ~self._committed_wrong
            rewards[a] = (
                self.cfg.rew_progress * progress
                + self.cfg.rew_hazard * in_haz.float()
                + self.cfg.rew_hazard_entry * entered.float()
                + self.cfg.rew_time
                + self.cfg.rew_success * bonus.float()
            )
            if self.cfg.route_instruction and a == "navigator":
                rewards[a] = rewards[a] + (
                    self.cfg.rew_wrong_corridor * self.in_wrong_corridor(a).float()
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
        if self.cfg.route_instruction and self.cfg.route_abort_wrong:
            # Obedience as a success CONDITION, not a per-step tax. A running
            # cost on corridor occupancy taught the policy that corridors are
            # where bad things happen and that standing still avoids them:
            # obedience climbed to 0.58 while completion of the route it already
            # knew fell 0.90 -> 0.03, and the same parking failure appeared
            # under a penalty alone and under two different shaping fields.
            # Aborting means driving is never punished; only committing to the
            # wrong corridor ends the episode, forfeiting the goal bonus. No
            # state is left in which stopping beats moving.
            #
            # It rides the time_out channel on purpose: the trainer reads
            # `terminated` as its success flag, so an abort must end the episode
            # without being scored a success, and dones zero the value
            # bootstrap either way.
            time_out = time_out | self.in_wrong_corridor("navigator")
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
        # collection / the navigator spawn curriculum), zero velocity
        def random_pose_rows(pose, rows):
            k = len(rows)
            if self.cfg.spawn_dist_range is not None:
                d_lo, d_hi = self.cfg.spawn_dist_range
                xy = sample_free_positions_band(
                    self._grid, self.cfg.cell, k, self._spawn_rng,
                    self._dfield["navigator"].cpu().numpy(),
                    self._dfield_origin, self._dfield_res, d_lo, d_hi,
                )
            else:
                xy = sample_free_positions(self._grid, self.cfg.cell, k, self._spawn_rng)
            yaw_r = self._spawn_rng.uniform(-math.pi, math.pi, size=k)
            pose[rows, 0] = origins[rows, 0] + torch.tensor(xy[:, 0], dtype=torch.float32, device=dev)
            pose[rows, 1] = origins[rows, 1] + torch.tensor(xy[:, 1], dtype=torch.float32, device=dev)
            pose[rows, 3] = torch.tensor(np.cos(yaw_r / 2), dtype=torch.float32, device=dev)
            pose[rows, 6] = torch.tensor(np.sin(yaw_r / 2), dtype=torch.float32, device=dev)

        # Decide the hazard corridor (and hence the route command, always the
        # safe side) BEFORE writing poses, so a mouth spawn can be placed at the
        # commanded corridor rather than a corridor picked independently.
        if self.cfg.force_slab_top is None:
            side_top = torch.rand(n, device=dev) < 0.5
        else:
            side_top = torch.full(
                (n,), bool(self.cfg.force_slab_top), dtype=torch.bool, device=dev
            )
        self._slab_top[env_ids] = side_top
        self._route_top[env_ids] = ~side_top

        for a in self.cfg.possible_agents:
            pose = torch.zeros(n, 7, device=dev)
            x, y, yaw = self._geo.starts[a]
            pose[:, 0] = origins[:, 0] + x
            pose[:, 1] = origins[:, 1] + y
            pose[:, 3] = math.cos(yaw / 2)
            pose[:, 6] = math.sin(yaw / 2)
            if self.cfg.randomize_spawns:
                random_pose_rows(pose, torch.arange(n, device=dev))
            elif a == "navigator" and self.cfg.spawn_curriculum_prob > 0.0:
                m = torch.tensor(
                    self._spawn_rng.random(n) < self.cfg.spawn_curriculum_prob,
                    device=dev,
                )
                self.curriculum_spawn[env_ids] = m
                if m.any():
                    random_pose_rows(pose, m.nonzero(as_tuple=True)[0])
            elif a == "navigator" and self.cfg.spawn_route_prob > 0.0:
                m = torch.tensor(
                    self._spawn_rng.random(n) < self.cfg.spawn_route_prob, device=dev
                )
                self.curriculum_spawn[env_ids] = m
                if m.any():
                    rows = m.nonzero(as_tuple=True)[0]
                    want_top = self._route_top[env_ids][rows]
                    for take_top in (True, False):
                        sel = rows[want_top == take_top]
                        if not len(sel):
                            continue
                        poses = self._route_poses[take_top]
                        k = int(round(self.cfg.spawn_route_frac * (len(poses) - 1)))
                        px, py, pyaw = poses[k]
                        pose[sel, 0] = origins[sel, 0] + px
                        pose[sel, 1] = origins[sel, 1] + py
                        pose[sel, 3] = torch.cos(pyaw / 2)
                        pose[sel, 6] = torch.sin(pyaw / 2)
            elif a == "navigator" and self.cfg.spawn_mouth_prob > 0.0:
                # corridor mouths, just inside the west entrance, facing east
                # (yaw 0 is the canonical heading, so quat stays identity)
                m = torch.tensor(
                    self._spawn_rng.random(n) < self.cfg.spawn_mouth_prob,
                    device=dev,
                )
                self.curriculum_spawn[env_ids] = m
                if m.any():
                    rows = m.nonzero(as_tuple=True)[0]
                    if self.cfg.route_instruction:
                        # Spawn at the mouth it was TOLD to use, so obedience and
                        # traversal are learned together from a state where both
                        # are achievable. A mouth drawn independently of the
                        # command would spend half its samples teaching the
                        # policy to drive the corridor it must not enter.
                        top = self._route_top[env_ids][rows]
                    else:
                        top = torch.tensor(
                            self._spawn_rng.random(len(rows)) < 0.5, device=dev
                        )
                    pose[rows, 0] = origins[rows, 0] - 2.5
                    pose[rows, 1] = origins[rows, 1] + torch.where(
                        top,
                        torch.tensor(1.0, device=dev),
                        torch.tensor(-1.5, device=dev),
                    )
                    pose[rows, 3] = 1.0
                    pose[rows, 6] = 0.0
            if a == "navigator" and self.cfg.spawn_yaw_jitter > 0.0:
                yaw = 2.0 * torch.atan2(pose[:, 6], pose[:, 3])
                yaw += torch.tensor(
                    self._spawn_rng.uniform(
                        -self.cfg.spawn_yaw_jitter, self.cfg.spawn_yaw_jitter, size=n
                    ),
                    dtype=torch.float32, device=dev,
                )
                pose[:, 3] = torch.cos(yaw / 2)
                pose[:, 6] = torch.sin(yaw / 2)
            pose[:, 2] = ROBOT_SIZE[2] / 2
            robot = self.scene[a]
            robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # teleport every slab cell to the side drawn above
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

        # re-seed the shaping potential and hazard state from the fresh poses.
        # Seeding _prev_in_hazard from the ACTUAL pose (not False) means a
        # robot that spawns inside the slab is not charged a phantom "entry".
        for a in self.cfg.possible_agents:
            self._prev_dist[a][env_ids] = self._shaping_dist(a)[env_ids]
            self._prev_in_hazard[a][env_ids] = self._in_hazard(a)[env_ids]
        self._committed_wrong[env_ids] = False
