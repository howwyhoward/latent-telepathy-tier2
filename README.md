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

- **Phase 0 — feasibility spike** (`spike/`): DONE. 546M env-steps/GPU-day
  at 64 envs / 128 tiled cameras at 64x64 (RGB + semantic segmentation,
  DLSS off). RL-on-pixels is tractable; no fallback needed. The original spike
  reported 447M; re-run and logged 2026-08-10 (`runs/spike/fps_benchmark.log`),
  which is the number with a committed artifact.
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
       0.98/0.00). Fix: -0.05/step restores Tier 1's 20% ratio. Warm-starting
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
  Six race generations returned nulls, ending with v6: warm-started from the
  mouth-curriculum trunk, a noiseless ground-truth slab bit (`oracle`) scored
  0.44/0.53 against 0.51 for silence, at either discount. `spike/
  diag_exploration.py` found why, and it was not the encoder or the credit
  horizon. From the canonical start the navigator sampled the top corridor
  **0.00 of 128 episodes** at its trained noise, and still 0.00 at double it:
  reaching that mouth needs ~+1.3 of sustained lateral action over the ~30-step
  decision window, and under iid noise at σ = 0.55 the mean deviation across
  those steps has std sigma/sqrt(30) ~ 0.10 — a 13-sigma event. Every race so far trained
  on batches containing no alternative route, so no message could be recruited.
  Tier 1 never met this: one gridworld action moved a whole cell, making route
  exploration and action exploration the same thing.
  v7 restores that temporal abstraction as *exploration only*: AR(1) noise
  (tau=30) on the lateral axis for the first 40 steps, which has marginal
  N(mu, sigma) at every step and so leaves PPO's per-step ratios exact,
  provided actions are re-scored under the std they were drawn with. Measured
  0.22 top-corridor coverage at 0.51 success (0.55 unperturbed); whole-episode
  noise also reached 0.12 but crushed success to 0.10. The boost decays to the
  trained std over 60% of training so the headline numbers end on-policy.
  v7 result: oracle 0.48, none 0.35, z_t 0.47 — null on the headline, but the
  most informative run of the series. Coverage worked and was then optimized
  away: slab-bottom success ran at 0.18-0.23 over iterations 1-40 and decayed
  to ~0.00 by iteration 80 while the boost was still near full strength, as
  slab-top success climbed 0.72 -> 0.96. The policy found the other corridor and
  learned to steer back out of it. Cause is intrinsic to Gaussian exploration:
  d logpi/dmu = (a-mu)/sigma^2, so the sigma that buys coverage attenuates the
  gradient carrying it — 0.22 per-sample in-window vs 1.67 post-window, over 40
  steps vs 560, i.e. ~100x more aggregate gradient saying "keep executing the
  bottom route" than "the route choice mattered".
  Two diagnostics then located the real blocker. `spike/
  diag_msg_sensitivity.py` (Isaac-free: one neighbour means the attention
  softmax is exactly 1, so pooled = v(msg_proj(msg)) independent of the image)
  shows the v7 oracle IS strongly message-driven, |dmu_y|/sigma_y = 2.69, vs
  0.23 for the v6 gamma-0.99 oracle and 0.74 for its gamma-0.999 variant
  (longer credit horizon recruited the bit partially; the exploration window
  did the rest). Measurements: runs/diag/msg_sensitivity.log.
  `spike/diag_route_choice.py` shows what it does with it, and it is not
  routing: deterministic rollouts take the bottom corridor in 100% of all four
  (slab side x true/lied bit) cells. Feeding a LIE moves slab-top success
  1.00 -> 0.41 without moving the corridor. So the bit was recruited to gate
  advance-vs-balk inside the chosen corridor, never to select the corridor —
  the refusal optimum of v2, now message-conditioned. Execution is not the
  problem (slab-top success is a clean 1.00) and neither is plumbing.
  Next: the corridor choice needs to be a first-class decision variable — a
  2-way categorical route head sampled once per episode — so exploration of it
  is Categorical (coverage without the 1/sigma^2 penalty) and it cannot be
  outvoted by the constant-view ego bias. This restores Tier 1's Discrete(5)
  property, the one architectural difference never revisited.
- **Phase 3.5 — route obedience** (`rl/train_route_obey.py`): before the head
  can choose a route, an executor must exist that can be told one. A 1-bit
  route command is appended to the receiver's feature vector (route-blind at
  load: widened layers keep their extra columns at zero) and trained until it
  is obeyed. Two mechanisms survived five failed recipes: obedience as a
  SUCCESS CONDITION (wrong-corridor entry ends the episode; per-step penalties
  instead taught "corridors are dangerous" and produced parking), and
  route-conditioned geodesic shaping (the field routes via the commanded
  corridor; the wrong corridor is a dead-end pocket whose gradient points back
  out, slopes bounded so no region outpays the correct route). Gate passed on
  canonical spawns, both directions: obedience 0.96/0.985, success 0.935/0.895
  (`runs/route_obey_v6/cont.pt`); with ±0.5 rad spawn-yaw jitter 0.97/0.91 and
  0.915/0.835 (`cont_yaw2.pt`). Composition check
  (`spike/eval_pixels_to_route.py`): scout pixels -> frozen JEPA latent ->
  supervised logistic probe -> route command -> frozen executor = 255/256
  success (0.996), 0 hazard steps, decode accuracy 1.000. (An earlier,
  unlogged pass of the same eval read 256/256; the number reported here is
  the one whose log is committed: runs/route_obey_v6/eval_pixels_to_route.log.)
- **Phase 3 closed — race v8, reward-only recruitment**
  (`rl/train_race_route.py`): executor FROZEN, the only learner a ~4.5k-param
  route head (message -> 2 logits), one categorical decision per episode
  committed at step 2 (reset-time camera frames are stale), credited with the
  episode's return — a contextual bandit, which is exactly the exploration
  structure v1-v7 lacked. Conditions on the same anchored 66-float wire:
  oracle (ground-truth bit, ceiling), z_t (scout's frozen JEPA latent, the
  thesis), none (zeros, floor). Route-optimality over 5 seeds:
  **z_t 0.986-1.000 (mean 0.997 ± 0.006 sd), oracle 0.982-0.996,
  none (anchored) 0.472-0.522 (mean 0.496 ± 0.020 sd)**; z_t hazard 0.00,
  executor obedience 1.000. Nobody told the head what the latent means — task
  reward alone recovered the slab bit from it. The original 3-seed zero-wire
  `none` (0.488-0.518) is superseded by the anchored floor; they agree to
  within noise.
  Post-mortems worth keeping: Bessel-corrected std() on a 1-element remainder
  minibatch NaNs the update (three runs died before it was traced), and a
  bandit head wants lr ~3e-3, not the pixel-policy 3e-4 (which reads as
  "not learning" at 112 Adam steps).
- **WP1 — corruption controls** (`spike/eval_race_head.py`, artifacts in
  `runs/diag/eval_race_head_*`): frozen v8 heads, GREEDY argmax decisions,
  per-env episode quotas (a completion-stream counter is length-biased:
  correct episodes finish ~206 steps, wrong ~380+, inflating corrupted modes
  to 0.55-0.64 before the fix). Intact wire: route-optimality **1.000** on
  all 3 z_t seeds and the oracle. Zeroed content, zeroed wire, shuffled
  sender, Gaussian noise: all at chance (0.41-0.56). Worst hazard 0.88
  steps — a balk-clip, never a ~20-step crossing. The decision lives in the
  message content, and only there.
- **WP2 — the full condition sweep** (`runs/race_v8b/`, same protocol,
  figure `plots/race_v8/v8b_sweep_bars.png`). Headline at 5 seeds:
  **z_t 0.986-1.000, oracle 0.982-0.996, none (anchored, zero content)
  0.472-0.522**. Motion-state steelmen at the floor 3/3 seeds each
  (position 0.480-0.490, kinematic 0.460-0.496) — the stationary scout's
  pose carries nothing; what it SEES carries everything. z_hat (predicted
  latent, Tier 1's C2) 0.992-1.000. raw_obs (12290-d unmatched wire):
  0.948/0.956/0.554 — one seed's entropy collapsed to ~1e-6, peaked at 0.83
  and fell back to chance, unrecoverable under lr annealing; the 66-float
  latent went 5/5. Compression is not just radio-feasible, it is easier to
  recruit.
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
