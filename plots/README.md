# Figure index

Every figure regenerates from committed artifacts — `python rl/plot_jepa.py`
and `python rl/plot_v8.py` rebuild the whole set. Same visual language as
Tier 1 (thesis condition red, controls blue/gray, per-seed dots over
translucent mean bars).

## top level — report opener (`rl/plot_fig1.py`)

| Figure | Shows | Data |
|---|---|---|
| `fig1_substrate.png` | Report Figure 1: the seed-2 chokepoint (slab in the scout's corridor) three ways. Tier 1 grid with BOTH agents' literal shadowcasting FOV — hazard inside the scout's, outside the navigator's; the Tier 2 scout camera with the slab filling a third of the frame (89,475/262,144 hazard px); the probe camera at the slabbed corridor's mouth with the slab fully hidden behind the staggered baffles (0/262,144). Same map all three panels — identical information structure, different substrate. | Tier 1 `envs/` (map + FOV) + `spike/out/hires/*_s2.png` (occlusion gate re-run at 512², seed 2: all gates PASS) |

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
