# Figure index

Every figure regenerates from committed artifacts. `rl/plot_jepa.py`,
`rl/plot_v8.py`, `rl/plot_diagnostics.py`, `rl/plot_fig3_occlusion.py` and
`rl/plot_fig2b_throughput.py` rebuild the whole set without touching Isaac.
Same visual language as Tier 1 (thesis condition red, controls blue/gray,
per-seed dots over translucent mean bars).

## top level — report opener (`rl/plot_fig1.py`)

| Figure | Shows | Data |
|---|---|---|
| `fig1_substrate.png` | Report Figure 1, 2×3: rows are the two substrates, columns are three viewpoints (the world / the corridor mouth where the route is chosen / the scout's post), so each column holds the viewpoint fixed and varies only the substrate. Hazard readouts per panel: Tier 1 cells-in-view 0/8 and 8/8 from `compute_visible`; Tier 2 hazard pixels 0 and 89,475 of 262,144 from the gate. Seed 2 (slab in the scout's corridor) throughout. | Tier 1 `envs/` (map + FOV) + `spike/out/hires/*_s2.png` (gate re-run at 512², all PASS) + `spike/out/hires/overhead_s2.png` (`spike/render_overhead.py`) |

## `diagrams/` — conceptual figures in Tier 1's figure language

| Figure | Shows | Data |
|---|---|---|
| `fig_tier2_pipeline.png` | The race v8 pipeline as a closed loop: scout camera → frozen shared encoder → 66-float wire → the 4,483-parameter route head → frozen executor → cmd_vel → the world, and the episode return dashed back into the head and nowhere else. The two paths that do not exist (head has no ego input; executor built with `broadcast_dim=0`) are stated beneath the head rather than floating in the dataflow lane. | `rl/plot_fig_pipeline.py`; numbers checked against `rl/train_race_route.py` and `chokepoint/route_head.py` |
| `fig_tier2_conditions.png` | Report Figure 2: one channel, six message contents. Left, the content-controlled channel with the shared 66-float wire drawn as 2 anchor floats + 64 content floats and the three held-fixed properties. Right, the ladder ordered by information content — floor, position, kinematic, z_t (C1), ẑ (C2), raw-pixel ceiling — each row carrying its measured route-optimality; the oracle hangs below the rule as an off-ladder diagnostic. Row-6 payload swatch is the scout's real frame at the encoder's 64×64 input resolution. | `rl/plot_fig2_conditions.py`; markers read from `runs/race_v8b/*.json` + `runs/race_v8/{z_t,oracle}*.json` (same pooling as the WP2 sweep) |

## `diagnostics/` — Phases 0–3: the instrument, and the six-null chain

| Figure | Shows | Data |
|---|---|---|
| `fig2b_throughput.png` | Report Figure 2b. Left: env-steps/sec vs num_envs against a linear-scaling reference — 546M env-steps/GPU-day at 64 envs / 128 tiled cameras, with the simulation rate falling only 135 → 99 steps/s across a 64× increase in cameras, so rendering is amortized. Right: the paired RGB / segmentation frame at the adopted configuration with per-class pixel counts, plus the three observed `idToLabels` orderings that make the lazy-table gotcha concrete. | `rl/plot_fig2b_throughput.py` from `runs/spike/fps_benchmark.log` + `spike/out/{rgb,seg}_e64_r64.npy` |
| `fig3_occlusion_gate.png` | Report Figure 3. Ten panels, RGB above the hazard mask the gate actually counts: navigator at start (0 px), scout with the slab in its own corridor (1,398 px), scout with the slab in the other corridor (0 px — absence is signal), the choice-point probe with baffles (0 px), and the same probe with baffles removed (**10 px, FAIL**). The pre-baffle leak is reproduced rather than quoted. Frames are the encoder's own 64×64, upscaled nearest-neighbour. | `rl/plot_fig3_occlusion.py` from `runs/gate/occl_{s2,s0,s2_nobaffle}.json` + `spike/out/occl_*` (`spike/verify_occlusion.py --no_baffles --tag --json_out`) |
| `fig4_m7_positive_control.png` | Report Figure 4. Success against the 0.80 gate, hazard steps/episode against the shaded 21–25 no-comms floor, episode length, and the slab-side split showing m7c's one-sided refusal (0.98 / 0.00) beside m7e's balance (0.92 / 0.98). Four runs overlaid. | `rl/plot_diagnostics.py` from `runs/archive/m7_navsolo/m7*.csv` |
| `fig7_exploration_collapse.png` | Report Figure 7, the methods contribution, built to read standalone in three steps. (1) The scene from above: the corridor always taken (128/128) against the one never taken (0/128), and the 1.25 m / ~3 s lateral cost of switching. (2) The sideways command averaged over the 30-step decision window, independent noise against AR(1) at the policy's own σ = 0.55 — ±0.10 (13× short of +1.3) against ±0.47 (still 2.8× short), so correlation is necessary but not sufficient. (3) The five measured settings, from the one all six races used (0.00 / 0.68) to persistent + lateral-only + first 40 steps (0.22 / 0.51). | `rl/plot_diagnostics.py` from `runs/diag/exploration.log`, `runs/diag/exploration_win.log` |
| `fig7b_exploration_sweep.png` | The σ-sweep parametrics behind Figure 7, kept for a reader who wants the curves rather than the four-point story: coverage and success against σ, the required deviation in σ under iid against AR(1) as a function, and the windowed/axis-restricted scatter. Not referenced in the report body. | `rl/plot_diagnostics.py` from `runs/diag/exploration.log`, `runs/diag/exploration_win.log` |
| `fig8b_composition_check.png` | Report Figure 8b. The §8.5 composition check drawn as the chain it was: scout pixels → frozen JEPA → **supervised** logistic probe → route bit → frozen v6 executor → navigation, with the one non-frozen box marked in red and each link's measurement hung beneath it (held-out probe 1.000, in-loop decode 1.000, obeyed 1.000, success 0.996 = 255/256, hazard 0.00). Below: success against the Stage 1.5 gate for the same executor handed the route directly (0.935 / 0.895, canonical spawns — a different episode set, shown for scale), and hazard steps against the ~20 a blind policy pays. Sets up race v8, which replaces the probe's labels with reward. | `rl/plot_fig8b_composition.py` from `runs/route_obey_v6/eval_pixels_to_route.log` |
| `fig8_v7_recruited_misused.png` | Report Figure 8. Left: v7's slab-bottom success decaying to 0.00–0.06 by iteration 80 while the exploration boost is still σ = 2.2 (3.6× the learned noise), schedule on a second axis. Right: the lie test — route unchanged at 100% bottom in all four cells while success falls 1.00 → 0.41. | `rl/plot_diagnostics.py` from `runs/race_v7/*.csv`, `runs/diag/route_choice_v7oracle.log` |

## `jepa/` — Phase 2: the frozen encoder (`rl/plot_jepa.py`)

| Figure | Shows | Data |
|---|---|---|
| `jepa_training.png` | Invariance loss (train/val) + effective rank of z over pretraining. Val tracks train; rank grows to ~44/64 — no collapse. | `runs/archive/jepa_v1/jepa_v1.csv` |
| `jepa_probes.png` | Held-out probe accuracy on the frozen latent vs majority baselines (hazard side, goal side; wall-distance R² annotated). | `checkpoints/jepa_pixels.pt` |

## `stage15/` — Phase 3.5: route obedience (`rl/plot_v8.py`)

| Figure | Shows | Data |
|---|---|---|
| `obey_gate_curves.png` | Canonical-spawn obedience + success for both commanded routes crossing the Stage-2 gate (0.90 / 0.80). | `runs/route_obey_v6/cont.json` |

## `race_v8/` — Phase 3: the recruitment result (`rl/plot_v8.py`)

| Figure | Shows | Data |
|---|---|---|
| `v8_race_curves.png` | Route-optimality learning curves, 3 seeds per condition: z_t rises to the oracle, none pins at 0.5. | `runs/race_v8/*.json` |
| `v8_race_seed_bars.png` | Final route-optimality per condition with per-seed dots. | `runs/race_v8/*.json` |
| `v8_success_seed_bars.png` | Final task success per condition. | `runs/race_v8/*.json` |
| `v8_hazard_bars.png` | Pre-registered readout: hazard steps/episode. | `runs/race_v8/*.json` |
| `v8_entropy_curves.png` | Decision entropy: message conditions commit, the floor cannot. | `runs/race_v8/*.json` |
| `v8_corruption_bars.png` | WP1 controls: frozen heads, greedy decisions — intact 1.000, every corruption (zero content / zero wire / shuffle / noise) at chance. | `runs/diag/eval_race_head_*.json` |
| `v8b_sweep_bars.png` | WP2 seven-condition sweep: z_t/z_hat/oracle saturate, none/position/kinematic at the floor, raw_obs optimization-fragile (1/3 seeds collapsed, annotated). | `runs/race_v8b/*.json` + v8 z_t/oracle |

## `demo/` — qualitative

| File | Shows |
|---|---|
| `v8_live_demo.mp4` | The composed system (scout pixels → JEPA → route head → frozen executor), overhead cinematic + science panel, slab side alternating across episodes. Regenerate: `spike/record_video.py`. Git-ignored (regenerable). |
| `v8_still_ep1.png`, `v8_still_ep2.png` | Stills from the two slab configurations. |

## Pending

WP3 (richer-than-one-bit scene) figures will form a new family here once
that experiment series exists. Every figure the report cites is rendered.
