# Experiment index

One directory per experiment series. Each run inside a directory produces
`<name>.log` (console), `<name>.csv` (per-iteration metrics), `<name>.pt`
(final checkpoint), `<name>.json` (summary), and the series' `launch.log`
records the exact launch flags.

## Active / load-bearing

| Directory | What it tested | Outcome |
|---|---|---|
| `race_v8b/` | **WP2 — the complete condition sweep** under the v8 protocol (frozen executor, bandit head, 6000 eps, lr 3e-3 annealed). New: `none` floor fixed (real anchor, zero content), `position`, `kinematic` (motion-state steelmen; the scout is stationary so both are constant wires), `z_hat` (predicted latent, Tier 1's C2), `raw_obs` (full 64x64x3 frame + anchor, 12290-d wire, deliberately unmatched), plus seeds 4–5 for `oracle`/`z_t`/`none`. | Route-optimality — **z_t 0.986–1.000 (5 seeds), z_hat 0.992–1.000, oracle 0.982–0.996 (5 seeds)** vs floors: none 0.472–0.522 (5 seeds), position 0.480–0.490, kinematic 0.460–0.496. **WP2 gate PASS: motion-state baselines at the floor 3/3 seeds.** `raw_obs`: 0.948/0.956/**0.554** — one seed's decision entropy collapsed to ~1e-6 by ep 2300, peaked 0.83, fell to chance and froze (787k params on 12290 inputs commit prematurely; the 66-float latent is 5/5). Compression is not just radio-feasible — it is easier to recruit. |
| `race_v8/` | **The recruitment result.** Frozen stage-1.5 executor (`cont.pt`); the only learner is a ~4.5k-param route head (message → corridor, one categorical decision/episode, credited with episode return — a contextual bandit). Conditions: `oracle` (ground-truth bit, ceiling), `z_t` (scout's frozen JEPA latent, thesis), `none` (zero wire, floor). | 3 seeds, route-optimality: **oracle 0.990–0.996, z_t 0.986–1.000, none 0.488–0.518**; z_t hazard 0.00, executor obedience 1.000 throughout. Reward-only decoding of the latent, within noise of the oracle. Two NaN post-mortems live in the trainer comments: reset-frame staleness and Bessel-corrected std on 1-element minibatches. |
| `route_obey_v6/` | Stage 1.5 continuation from the v4 `abort_mouth` checkpoint. `cont` = plain continuation; `cont_yaw` = +0.5 rad spawn-yaw jitter from the weak checkpoint (plateaued); `cont_yaw2` = yaw jitter warm-started from the **gate-passing** `cont.pt`. | **`cont` PASSED the Stage-2 gate**: canonical obedience 0.96/0.985, success 0.935/0.895 (and `cont_yaw2` passed under jitter: 0.97/0.91, 0.915/0.835). `eval_pixels_to_route.log` = the composed pixels→JEPA→probe→route→policy system: 255/256 success (0.996), 0 hazard steps, decode 1.000. |
| `race_v7/` | Stage 2 race with windowed, axis-restricted, annealed AR(1) exploration noise. Conditions: `oracle`, `z_t`, `none`. | Coverage achieved (~22% top-corridor sampling) but 1/σ² gradient attenuation meant the policy never consolidated the alternative route. Motivated the Stage 1.5 decomposition. |
| `nav_pretrain/` | Stage 1 navigation pretrain without messages. `nav_s1` = uniform random spawns (failed, 0.00), `nav_s1_band` = banded curriculum (collapsed as band widened), `nav_s1_mouth` = fixed canonical + corridor-mouth spawns. | `nav_s1_mouth.pt` reached ~0.72 success from canonical start and became the warm-start trunk for all route-obedience work. |
| `diag/` | One-off diagnostics: exploration coverage sweeps, v7-oracle route-choice causal test, msg-sensitivity measurements, and the **WP1 corruption eval** (`eval_race_head_*`): frozen v8 heads, greedy argmax decisions, five wire modes (intact / zero_content / zero_all / shuffle / noise), equal-weight per-env episode accounting. | v7: oracle message WAS recruited but only gated advance-vs-balk, not corridor choice. WP1: **intact 1.000 route-optimality on all 3 z_t seeds + oracle; every corruption mode collapses to 0.41–0.56 (chance); executor obedience 1.000, worst hazard 0.88 steps (balk-clip, vs ~20 for a crossing). All pre-registered checks PASS.** First run diagnosed a length-biased completion-stream estimator (fast correct episodes over-collected, corrupted modes read up to 0.64); fixed by per-env quotas. |

## Route-obedience iterations (Stage 1.5 history)

| Directory | Recipe | Why it failed / what it taught |
|---|---|---|
| `route_obey/` (v1) | Per-step wrong-corridor penalty | Policy parked outside corridors: penalties taught "corridors are dangerous". |
| `route_obey_v2/` | Tuned penalties + first-entry metric | Entered the commanded corridor then stopped ("parking"): optimized the entry metric, not traversal. |
| `route_obey_v3/` | Corrected route-conditioned geodesic fields (dead-end pocket, bounded slopes, clamped shaping) | Fields fixed the "bulldozing" exploit; traversal still weak from canonical start. |
| `route_obey_v4/` | Obedience as success condition (wrong corridor = episode failure) + 50% commanded-mouth spawns | Best recipe: `abort_mouth.pt` showed a genuine upward canonical trend. |
| `route_obey_v5/` | Reverse curriculum along the commanded route | First non-zero canonical obedience; isolated the initial-heading gap. |
| `route_obey_v6/` | v4 recipe continued (see above) | **Gate passed.** |

## Instrument measurements (Phase 0–1, re-run for the figures)

| Directory | Contents |
|---|---|
| `spike/fps_benchmark.log` | Phase 0 throughput, num_envs ∈ {1, 4, 16, 64} at 64×64: 546M env-steps/GPU-day at 64 envs, simulation rate 135 → 99 steps/s across a 64× camera increase. Also captures three different `idToLabels` orderings, which is the lazy-table gotcha as evidence. Figure 2b. |
| `gate/occl_s2.json` | Occlusion gate, seed 2 (slab TOP): navigator 0 px, scout 1,398 px, both choice points 0 px. PASS. |
| `gate/occl_s0.json` | Occlusion gate, seed 0 (slab BOTTOM): all four probes 0 px — the scout's *absence* reading. PASS. |
| `gate/occl_s2_nobaffle.json` | The same scene with `--no_baffles`, reproducing the pre-baffle straight corridor: choice point leaks **10 px**. FAIL, and that failure is the measurement the baffles were designed against. Figure 3. |

## Archive

| Directory | Contents |
|---|---|
| `archive/m7_navsolo/` | M7 milestone: first single-agent PPO in the Tier-2 world (m7–m7e). Proved geodesic shaping > Euclidean, episode length 60 s, retrain-from-scratch > warm-start after reward changes. |
| `archive/jepa_v1/` | JEPA pretrain log/CSV and the data-collection log. The trained encoder lives in `checkpoints/jepa_pixels.pt`; linear probes read slab side at 100%. |
| `archive/misc/` | Streaming-viewer and smoke-test session logs. |

`race/`–`race_v6/` are earlier Stage-2 attempts (refusal optimum, rung exploit,
gamma/credit-assignment fixes, spawn curricula); kept for the negative-results
record.
