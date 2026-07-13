# Latent Telepathy — Tier 2 (Sim-to-Real, Isaac Lab)

Confirmatory tier: show the Tier 1 mechanism (broadcast frozen JEPA latents,
not kinematics) survives physics-accurate pixel feeds, then bridge to the
Wu Lab RoboMaster testbed. The evidentiary core lives in Tier 1
(`~/latent-telepathy`); this repo is the Isaac Lab fork.

## Machine layout (wulab1)

The root disk is effectively full, so all heavy artifacts live on `/data`:

| What | Where |
| --- | --- |
| Project code (this repo) | `~/latent-telepathy-tier2` |
| Conda env (Python 3.11) | `/data/howard/isaac/envs/isaaclab` |
| Isaac Lab clone | `/data/howard/isaac/IsaacLab` |
| Omniverse caches (shaders, assets) | `/data/howard/isaac/cache` (symlinked from `~/.cache/ov` etc.) |

`isaac-data/` in this repo is a symlink to `/data/howard/isaac`.

## Daily usage

```bash
source setup/env.sh          # activates env, pins GPU 1, routes caches to /data
```

Run everything long-lived inside `tmux` so it survives SSH disconnects.

## Known constraints

- **Driver:** server runs 550.90.12; Isaac Sim 5.1 is validated on 580.65.06.
  Older-than-validated usually works but is unsupported — if the RTX renderer
  refuses to start, fall back to Isaac Sim 4.5 + Isaac Lab 2.1 (Python 3.10)
  and ask the admin to upgrade the driver to 580.65.06.
- **GPUs:** 0 has a fan-sensor error, 2–3 run labmates' jobs. We use GPU 1.
- **No GUI on this box:** use `--headless` always; for interactive inspection
  use the WebRTC streaming client on the Mac (`LIVESTREAM=2`).

## Phase plan (from the Tier 2 planning discussion)

- **Phase 0 — feasibility spike** (`spike/`): DONE. 447M env-steps/GPU-day
  at 64 envs / 128 tiled cameras at 64x64 (RGB + semantic segmentation,
  DLSS off). RL-on-pixels is tractable; no fallback needed.
- **Phase 1 — instrument**: IN PROGRESS. The chokepoint scene is extruded
  directly from Tier 1's `generate_chokepoint_map()` (0.5 m/cell) in
  `spike/verify_occlusion.py`. Occlusion is verified empirically via static
  probe cameras at the corridor-choice points (hazard-class pixel counts in
  segmentation masks). Tier 1's FOV-radius occlusion has no 3D equivalent —
  a straight corridor leaked 10 hazard pixels to the choice point — so each
  corridor has two staggered light baffles (0.6 m stubs, north/south
  attached, columns 8 and 10) that block every straight ray from the mouth
  to the slab while leaving a 0.4 m S-gap passable by a 0.24 m robot.
  Peek-past-the-baffle remains possible, matching Tier 1's peek-and-reroute.
  Gates (navigator start == 0 px, scout consistent with slab side, both
  choice points == 0 px) pass on both slab placements (seeds 0, 2).

  The scene builder lives in `chokepoint/scene.py` and is shared by the gate
  and the RL env, so the certified geometry and the training geometry cannot
  drift apart. `chokepoint/env.py` is the `DirectMARLEnv` (PettingZoo-shaped:
  dict obs/actions per agent): cmd_vel Box(3,) actions matching
  `/robomaster_N/cmd_vel`, 64x64 RGB obs per robot, hazard slabs passable
  (no collider) with a geometric AABB penalty, slab side coin-flipped per env
  at reset by teleporting the slab prims, Tier 1-shaped reward skeleton
  (potential-based progress + hazard penalty + team success bonus).
  Smoke test (`spike/smoke_env.py`): obs shapes/ranges, per-env slab
  randomization, hazard flag, timeout resets — all pass.

  Message bus + receiver ported (`chokepoint/message_bus.py`,
  `chokepoint/receiver.py`, unit-tested Isaac-free in `tests/`): vectorized
  torch, Euclidean comm radius, anchored delivery, frozen-encoder +
  zero-init-value-path discipline kept from Tier 1; policy head is now a
  diagonal Gaussian over cmd_vel. Integration smoke (`spike/smoke_bus.py`,
  env -> bus -> receiver -> actions -> step) passes at num_envs=8,
  ~26 control-steps/s with per-agent python-loop forwards (unoptimized).

  Gotcha: torch's cuDNN 9 fails to initialize on driver 550
  (CUDNN_STATUS_NOT_INITIALIZED on the first conv). `chokepoint/__init__.py`
  disables cuDNN — plain CUDA kernels are fine at 64x64. Remove when the
  driver reaches >= 570. Crashes after Kit is up used to hang silently in
  extension teardown; smoke scripts install a sys.excepthook that os._exit(1)s.

  M7 positive control (`rl/ppo_pixels.py`, CleanRL-style PPO, end-to-end CNN
  on pixels, navigator only, scout parked): PASS at 0.94 success
  (`runs/m7e_*`), balanced across slab sides (0.92/0.98), ~21 hazard
  steps/episode = blind crossing on ~half the episodes. That hazard rate is
  the no-comms floor for the Phase 3 race. Three failed runs taught the
  design constraints, all documented in code comments:
    1. m7  (0.00): Euclidean shaping pins the robot into the second baffle's
       corner. Fix: geodesic Dijkstra field (`chokepoint/geometry.py`, with a
       stall-point regression test) + 60 s episodes.
    2. m7b (0.45): undertrained at 2M steps, still climbing at anneal end.
    3. m7c/d (0.55): hazard at -0.5/step made crossing cost 2x the success
       bonus — the policy rationally refused to cross (slab-side split
       1.00/0.00). Fix: -0.05/step restores Tier 1's 20% ratio. Warm-starting
       across the reward change did NOT work (the refusing policy never
       samples the slab, so it never sees the new price); m7e retrained fresh.
- **Phase 2 — JEPA on pixels**: DONE, all four pre-registered gates PASS.
  Dataset: 204,800 frames / 200k transitions from random-policy rollouts with
  uniform free-pose spawns (`spike/collect_jepa_data.py`; note: the sim's
  segmentation idToLabels table grows lazily, remap per step). Model
  (`chokepoint/jepa.py`) ports Tier 1's BYOL/EMA/VICReg recipe with a CNN
  encoder (64x64 RGB -> 64-D), continuous-action predictor, and a 16x16
  segmentation-logit decoder (hazard/goal/agent x10, decoded from both z_t
  and z_pred). Trainer `rl/train_jepa.py`. Results (`checkpoints/
  jepa_pixels.pt`, frozen): hazard-visible linear probe 0.935 (majority
  0.857), goal 0.911 (0.850), wall-count R^2 0.985, eff_rank 44.5/64,
  min_std 0.92 — no collapse.
- **Phase 3 — positive control, then reduced race** (4 conditions x 3 seeds).
- **Phase 4 — ROS2 deployment** on 2 RoboMasters (TensorRT encoder on Jetson,
  64-float latent topic).

## First-time verification

```bash
source setup/env.sh
# smoke test: import + headless kit launch (validates driver stack)
python spike/smoke_import.py
# Isaac Lab bundled env, headless
cd /data/howard/isaac/IsaacLab
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-v0 --num_envs 32 --headless
```
