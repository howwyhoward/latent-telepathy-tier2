# Figure index

Every figure regenerates from committed artifacts — `python rl/plot_jepa.py`
and `python rl/plot_v8.py` rebuild the whole set. Same visual language as
Tier 1 (thesis condition red, controls blue/gray, per-seed dots over
translucent mean bars).

## top level — report opener (`rl/plot_fig1.py`)

| Figure | Shows | Data |
|---|---|---|
| `fig1_substrate.png` | Report Figure 1, 2×3: rows are the two substrates, columns are three viewpoints (the world / the corridor mouth where the route is chosen / the scout's post), so each column holds the viewpoint fixed and varies only the substrate. Hazard readouts per panel: Tier 1 cells-in-view 0/8 and 8/8 from `compute_visible`; Tier 2 hazard pixels 0 and 89,475 of 262,144 from the gate. Seed 2 (slab in the scout's corridor) throughout. | Tier 1 `envs/` (map + FOV) + `spike/out/hires/*_s2.png` (gate re-run at 512², all PASS) + `spike/out/hires/overhead_s2.png` (`spike/render_overhead.py`) |

## `diagrams/` — conceptual figures in Tier 1's figure language

| Figure | Shows | Data |
|---|---|---|
| `fig_tier2_conditions.png` | Report Figure 2: one channel, six message contents. Left, the content-controlled channel with the shared 66-float wire drawn as 2 anchor floats + 64 content floats and the three held-fixed properties. Right, the ladder ordered by information content — floor, position, kinematic, z_t (C1), ẑ (C2), raw-pixel ceiling — each row carrying its measured route-optimality; the oracle hangs below the rule as an off-ladder diagnostic. Row-6 payload swatch is the scout's real frame at the encoder's 64×64 input resolution. | `rl/plot_fig2_conditions.py`; markers read from `runs/race_v8b/*.json` + `runs/race_v8/{z_t,oracle}*.json` (same pooling as the WP2 sweep) |

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
that experiment series exists.
