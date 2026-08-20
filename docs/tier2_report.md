# Tier 2 Report — Roadmap, Results, and the Path to Hardware

Part II of the Tier 2 documentation (sections 7–12; sections 1–6 cover
motivation, claims, conditions, and architecture). Every number in this
document traces to a committed artifact; sources are named per figure.

Updated 2026-08-19: WP1/WP2 remain the confirmatory program. Mechanical
corrections in this pass: dual-stack perception (the navigator encoder
crosses too), figure/table n aligned at 5 seeds, probe labels, and
number-drifts listed in §12.

---

## 7. Roadmap Part I — Building the Instrument (Phases 0–1)

> **[FIGURE 1 — Same information structure, different substrate]**
> `plots/fig1_substrate.png` — 2×3: rows = gridworld vs Isaac, columns =
> world / corridor mouth / scout's post. Seed 2 throughout. Source:
> `rl/plot_fig1.py`.

### 7.1 Phase 0 — Feasibility spike

Before committing to RL-on-pixels, measure whether the hardware can deliver
the sample budget. If it cannot, the tier restructures entirely.

Throughput was benchmarked with tiled cameras producing RGB and semantic
segmentation at encoder resolution, DLSS off, across several `num_envs`
settings.

**Result: 546M env-steps/GPU-day** at 64 environments / 128 tiled cameras at
64×64. Tractable. No fallback needed.

The original spike reported 447M; the benchmark was re-run on 2026-08-10 on an
otherwise idle box and logged, so the figure above is the one with a committed
artifact (`runs/spike/fps_benchmark.log`). Either number clears the bar by an
order of magnitude, and the decision it drove is unchanged.

The spike also dumped RGB and segmentation frames per configuration so the
semantic labelling could be confirmed wired correctly — the same check that
later becomes the occlusion gate.

> **[FIGURE 2b — Throughput and the semantic channel]**
> `plots/diagnostics/fig2b_throughput.png` — left: env-steps/sec vs num_envs
> against a linear-scaling reference, annotated with the GPU-day figure and the
> measured render cost (simulation rate falls only 135 → 99 steps/s across a
> 64× increase in cameras). Right: the paired RGB / segmentation frame at the
> adopted configuration, with per-class pixel counts, plus the three observed
> `idToLabels` orderings that make the lazy-table gotcha concrete. Source:
> `runs/spike/fps_benchmark.log`, `spike/out/{rgb,seg}_e64_r64.npy`,
> `spike/spike_fps_benchmark.py`.

### 7.2 Phase 1 — Scene and the occlusion gate

Occlusion is the load-bearing mechanic. In Tier 1 it was a tested geometric
algorithm. In Tier 2 it is an emergent property of 3D geometry and lighting,
so it must be measured on the actual renderer.

The gate places static probe cameras at both corridor-choice points and
counts hazard-class pixels in the segmentation masks. Three criteria,
pre-registered:

| Gate | Criterion |
|---|---|
| Navigator at start | hazard pixels == 0 |
| Scout at start | hazard pixels > 0 iff slab is in its corridor, == 0 otherwise |
| Both choice-point probes | hazard pixels == 0 — the strict gate |

The scout criterion is worth pausing on: **absence is signal**. The scout
distinguishes the two worlds by presence versus absence of hazard pixels from
a fixed vantage.

The straight corridor failed. With no baffles, 10 hazard pixels leaked from
the slab to the choice point — enough for the navigator to solve the task
from its own camera and silently void the entire experiment.

```
BEFORE — straight corridor, sightline leaks       AFTER — staggered baffles

┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
│ ███████████████████████████████████  │   │ ███████████████████████████████████  │
│                                      │   │            ▐▌                        │
│  ◄══════ sightline ══════════ ▓▓▓▓   │   │  ◄─ ─ ─ ─ ─▐▌X    ▐▌  blocked  ▓▓▓▓  │
│  ▲                            slab   │   │  ▲          ▐▌   ▐▌            slab  │
│  choice point: 10 hazard px          │   │  choice point: 0 hazard px           │
│ ███████████████████████████████████  │   │ ███████████████████████████████████  │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
                                            0.6 m stubs, cols 8 & 10, alternating
                                            sides. 0.4 m S-gap ≫ 0.24 m robot.
```

The baffles were designed against that measurement. All three gates then pass
on both slab placements. Peek-past-the-baffle remains possible — matching
Tier 1's peek-and-reroute — but requires committing to a corridor first,
which is precisely the decision the message is supposed to inform.

> **[FIGURE 3 — The occlusion gate, real renderer output]**
> `plots/diagnostics/fig3_occlusion_gate.png` — ten panels, RGB above the
> counted hazard mask: navigator at start, scout with the slab in its own
> corridor, scout with the slab in the other corridor (absence is signal), the
> choice-point probe with baffles, and the same probe with the baffles removed.
> Each carries its hazard-pixel count and gate verdict. Tier 2's analog of Tier
> 1's shadowcasting figure — literal tool output at the encoder's own 64×64, not
> a schematic. The pre-baffle leak is *reproduced*, not quoted: `--no_baffles`
> rebuilds the straight corridor and the gate fails at exactly 10 px. Source:
> `spike/verify_occlusion.py` (`--no_baffles`, `--tag`, `--json_out`),
> `runs/gate/occl_{s2,s0,s2_nobaffle}.json`, `spike/out/occl_*`.

### 7.3 M7 — the pixel positive control

Before any communication exists, prove PPO learns this task from these
pixels — so any later failure implicates the message, not the RL core. Three
failed runs each taught a design constraint.

| Run | Success | Diagnosis and fix |
|---|---|---|
| m7 | 0.00 | Euclidean shaping pins the robot into the second baffle's corner — the potential points through a wall. Fix: geodesic Dijkstra field with a stall-point regression test asserting the gradient at that exact corner points north; episodes extended to 60 s. |
| m7b | 0.45 | Undertrained at 2M steps, still climbing at anneal end. A budget problem, not a design problem. |
| m7c/d | 0.55 | Hazard at −0.5/step made crossing cost 2× the success bonus, so the policy rationally refused to cross — slab-side split 0.98 / 0.00. Fix: −0.05/step, restoring Tier 1's ~20% ratio. Warm-starting across the reward change did not work: a refusing policy never samples the slab, so it never observes the new price. Retrained fresh. |
| m7e | **0.94** | **PASS.** Balanced across slab sides (0.92 / 0.98), ~21–25 hazard steps/episode — blind crossing on roughly half the episodes, exactly as expected of a policy with no information about slab side. |

That hazard figure is the tier's **no-communication floor** — what a
competent, fully-trained, message-less navigator pays. It is the number z_t
later drives to 0.00.

> **[FIGURE 4 — M7 positive control]**
> `plots/diagnostics/fig4_m7_positive_control.png` — four panels in Tier 1's M7
> style: success rate against the 0.80 gate, hazard steps/episode against the
> shaded no-comms floor, episode length, and the slab-side split that exposes
> m7c's one-sided refusal next to m7e's balance. m7 / m7b / m7c / m7e overlaid,
> so the three failure modes and the pass read in one frame. Source:
> `runs/archive/m7_navsolo/m7*.csv`.

---

## 8. Roadmap Part II — Representation, Six Nulls, and the Decisive Test

### 8.1 Phase 2 — JEPA on pixels

The encoder's output is the message. It must be validated to carry hazard and
goal content before it is frozen, or every downstream null becomes
uninterpretable. This is the single most important sequencing decision
inherited from Tier 1, and it is why six subsequent nulls stayed diagnosable.

Dataset: 204,800 frames (1,600 steps × 128 streams) from random-policy
rollouts with uniform free-pose spawns; 175,504 training transitions, 4,096
held out.

Collection gotcha worth recording: the simulator's segmentation `idToLabels`
table grows lazily, so the class remap must be recomputed per step. A stale
table silently mislabels the rare classes that matter most.

Results — frozen checkpoint, all four gates PASS on the first trained model:

| Metric | Gate (as coded) | Result |
|---|---|---|
| Hazard-visible probe (linear / MLP) | > majority | 0.935 / 0.942 (majority 0.857) |
| Goal-visible probe (linear / MLP) | > majority | 0.911 / 0.927 (majority 0.850) |
| Wall-count R² | > 0.5 | 0.985 |
| Effective rank | > 0.3 × 64 | 44.5 / 64 |
| Min per-dim std | ≥ 1e-2 | 0.924 (mean 1.032) |

Effective rank climbed monotonically from 6.9/64 at step 200 to 44.4/64 at
step 3400 — VICReg doing visible work against the collapse that bit Tier 1
immediately.

**Honest reading of the hazard probe.** These are *visibility* probes on
uniform free-pose frames, not slab-side probes at the scout's post. 0.935
against a 0.857 majority is a 55% reduction in error, not a dramatic absolute
margin, and it is pooled accuracy only — no balanced accuracy was logged, the
same omission that later inflated the sim-to-real transfer number. One
encoder seed. The probe is a necessary check that hazard/goal pixels are
linearly present in z; the race is the arbiter of whether that content is
usable as a route bit — and the race's answer (0.997) is considerably
stronger than the probe alone would predict.

> **[FIGURE 5 — JEPA training health]** `plots/jepa/jepa_training.png` —
> invariance loss (train vs held-out val) and effective rank over training.
> Source: `runs/archive/jepa_v1/jepa_v1.csv`.

> **[FIGURE 6 — Latent information content]** `plots/jepa/jepa_probes.png` —
> visibility-probe accuracies vs majority baselines; wall-count R² annotated.
> Baselines visually prominent: the point of the figure is clearing them.
> Source: `checkpoints/jepa_pixels.pt` (probe_metrics), log in
> `runs/archive/jepa_v1/`.

### 8.2 Phase 3 — the diagnostic chain

Six generations returned nulls. The seventh returned a null that explained
all of them. The chain is presented in full because the reasoning is the
contribution.

```
v1  ── the world was too cheap ────────────────────────────────────────────
    none 0.99 @ 14.97 hazard steps · position 0.74 · z_t 0.94 · raw 0.82
    Every condition solved it by walking straight through the slab.
    → Raise the hazard price. Seal the rung.

v2–v6  ── the refusal optimum ─────────────────────────────────────────────
    Fixed price, rung, discount, spawn curriculum. Kept landing on:
    solve one slab side perfectly, refuse the other, sit at ~0.5.
    v6 is the diagnostic low point:
       a NOISELESS ORACLE BIT scored 0.44/0.53 vs 0.51 for silence.
    → A perfect message worth nothing is not a tuning problem.

DIAGNOSIS  ── route exploration ≠ action exploration ──────────────────────
    The alternative corridor was sampled 0 / 128 episodes at trained noise.
    Still 0 at double it.  Needs ~+1.3 sustained lateral over ~30 steps;
    at the measured grid point σ = 0.55, σ/√30 ≈ 0.10  ⟹  a 13σ event.
    Six nulls, one cause: no batch ever CONTAINED the alternative route.

v7  ── coverage achieved, then optimized away ─────────────────────────────
    AR(1) noise (τ=30), lateral axis, first 40 steps, σ = 4.48
    (explore_log_std = 1.5) → 0.22 coverage.
    Then: slab-bottom success 0.18–0.23 → 0.00 by iter 80,
          WHILE the boost had only annealed to σ = 2.2 (3.6× learned).
    ∂logπ/∂μ = (a−μ)/σ²  ⟹  the σ that buys coverage attenuates its
    own gradient. ~100× more aggregate gradient says "keep going straight."

v8  ── make the route a first-class decision ──────────────────────────────
    Freeze the executor. One categorical choice per episode. A bandit.
    → z_t 0.997 vs none 0.496.  ★
```

### 8.3 The exploration measurement

`spike/diag_exploration.py` measured, at the trained policy's own noise
level, how often the navigator ever sampled the alternative corridor:

| log_std | σ | τ | top | bottom | success |
|---|---|---|---|---|---|
| −0.60 (nearest grid; policy vy σ ≈ 0.60) | 0.55 | 0 | 0.00 | 1.00 | 0.68 |
| 0.00 (≈2× noise) | 1.00 | 0 | 0.00 | 1.00 | 0.58 |
| 0.50 | 1.65 | 30 | 0.16 | 0.60 | 0.11 |

Zero, and still zero at double the noise. Buying coverage by raising σ alone
destroys the policy long before it buys much.

**Why Tier 1 never met this.** One gridworld action moved a whole cell. Route
exploration and action exploration were the same operation. The temporal
abstraction that made Tier 1's exploration tractable was an accident of the
action space, and it did not survive the port.

> **[FIGURE 7 — The exploration collapse]**
> `plots/diagnostics/fig7_exploration_collapse.png`. The most important
> diagnostic figure in the tier, and the one that must read standalone, because
> it is the methods contribution. Read left to right as three steps. **(1) The
> decision:** the scene from above, the corridor the warm-started policy always
> takes (128/128) against the one it never takes (0/128), and the cost of
> switching — 1.25 m of lateral travel held for ~3 s, with the slab redrawn
> each episode so the never-taken route is sometimes the only safe one. **(2)
> Why chance never does it:** the sideways command averaged over the 30-step
> decision window, for noise drawn fresh each step against noise that persists
> for ~3 s, both at the policy's own σ = 0.55. Independent noise cancels down
> to ±0.10 and needs 13× that to reach +1.3; AR(1) noise lands at ±0.47 and is
> still 2.8× short, which is why correlation alone is necessary but not
> sufficient. **(3) What actually worked:** the five measured settings, from the
> one all six races used (0.00 coverage, 0.68 success) through louder noise and
> persistent-but-quiet noise (both still 0.00) to persistent-and-louder on every
> axis (0.16 coverage, but success collapses to 0.11) and finally persistent,
> lateral-only, first 40 steps (0.22 coverage at 0.51 success). The σ-sweep
> parametrics behind panels 2 and 3 are in
> `plots/diagnostics/fig7b_exploration_sweep.png`. Source:
> `runs/diag/exploration.log`, `runs/diag/exploration_win.log`.

### 8.4 Generation 7 — why coverage was not enough

v7 restored temporal abstraction as exploration only: AR(1) noise (τ = 30) on
the lateral axis for the first 40 steps. The construction is deliberately
PPO-safe — AR(1) has marginal N(μ, σ) at every step, so per-step ratios stay
exact provided actions are re-scored under the σ they were drawn with.

Measured 0.22 top-corridor coverage at 0.51 success — that is the
**start-of-schedule** diagnostic (`explore_log_std = 1.5`, σ = 4.48), not the
iter-80 value. Then the optimizer undid it: slab-bottom success peaked near
0.22 in the first ~20 iterations and decayed to 0.00–0.06 by iteration 80 —
while the exploration boost had annealed only to σ = 2.2, or 3.6× the
policy's own learned noise. The policy found the other corridor and learned
to steer back out of it.

The cause is intrinsic to Gaussian policy gradients. Since
∂logπ/∂μ = (a − μ)/σ², the σ that buys coverage attenuates the very gradient
that would carry the discovery: 0.22 per-sample in-window versus 1.67
post-window, over 40 steps versus 560 — roughly 100× more aggregate gradient
saying "keep executing the corridor you're in" than "the corridor choice
mattered." No entropy schedule fixes this; a schedule moves both terms the
same way.

Two diagnostics then located the blocker precisely.

1. **The channel was recruited.** `spike/diag_msg_sensitivity.py` measures
   message-driven action change analytically (with one neighbour, the
   attention softmax is exactly 1, so pooled context reduces to
   v(msg_proj(msg)), independent of the image). The v7 oracle policy scored
   |∂μ_y|/σ_y = 2.69 versus 0.23 for the v6 γ=0.99 oracle (0.74 for its
   γ=0.999 variant — a longer credit horizon recruited the bit partially).
   Measurements: `runs/diag/msg_sensitivity.log`.
2. **But recruited for the wrong function.** `diag_route_choice.py` found
   deterministic rollouts taking the bottom corridor in 100% of all four
   cells (slab side × true/lied bit). Feeding a lie moved slab-top success
   1.00 → 0.41 without moving the corridor.

So the bit was gating advance-versus-balk inside the already-chosen
corridor — the refusal optimum, now message-conditioned. Execution was not
the problem. Plumbing was not the problem. The corridor choice was never a
decision variable the optimizer could reach.

> **[FIGURE 8 — Recruited, but for the wrong function]**
> `plots/diagnostics/fig8_v7_recruited_misused.png`. Left: v7 coverage decay —
> slab-bottom success against iteration for all three conditions, with the
> exploration-boost schedule on a second axis. At iteration 80, where the
> alternative route has gone, the boost is still σ = 2.2, or 3.6× the learned
> noise. Right: the lie test as a 2×2 grid (slab side × true/lied bit) showing
> corridor choice unchanged at 100% bottom while success drops 1.00 → 0.41.
> Source: `runs/race_v7/*.csv`, `runs/diag/route_choice_v7oracle.log`.

### 8.5 Phase 3.5 — building the missing abstraction

Before a head can choose a route, an executor must exist that can be told
one. This is the decomposition v1–v7 lacked, and it restores Tier 1's
Discrete(5) property — the one architectural difference never revisited.

A 1-bit route command is appended to the receiver's feature vector — appended
rather than mixed into the ego embedding, so a trunk trained without it loads
unchanged and is initially route-blind (widened layers keep their extra
columns at zero).

Five recipes failed first. The two mechanisms that survived are the
interesting part.

| Iteration | Recipe | What it taught |
|---|---|---|
| v1 | per-step wrong-corridor penalty | Policy parked outside both corridors — penalties taught "corridors are dangerous," not "use that one." |
| v2 | tuned penalties + first-entry metric | Entered the commanded corridor, then stopped. Optimized the entry metric, not traversal. |
| v3 | corrected route-conditioned geodesic fields | Fixed a "bulldozing" exploit; canonical traversal still weak. |
| v4 | obedience as a success condition + commanded-mouth spawns | Best recipe. First genuine upward canonical trend. |
| v5 | reverse curriculum along the commanded route | First non-zero canonical obedience; isolated the initial-heading gap. |
| v6 | v4 recipe continued | **Gate passed.** |

The two surviving mechanisms:

1. **Obedience as a success condition, not a penalty.** A per-step penalty is
   a gradient on being somewhere. A terminal condition is a gradient on the
   decision. The former produced parking; the latter produced obedience.
2. **Route-conditioned geodesic shaping.** The potential field routes via the
   commanded corridor, making the wrong corridor a dead-end pocket whose
   gradient points back out, with slopes bounded so no region outpays the
   correct route. Five unit tests assert exactly these properties — downhill
   from the start, own-mouth preference, commanded-corridor descent, bounded
   slope, wrong-corridor pocket — which is why the field could be trusted
   when training results were ambiguous.

**Gate: PASS.** Canonical spawns, both directions: obedience 0.960 / 0.985,
success 0.935 / 0.895. (A constant corridor command would score 0.5
obedience.) Under ±0.5 rad spawn-yaw jitter: 0.97 / 0.91 obedience,
0.915 / 0.835 success.

> **[FIGURE — Stage 1.5 obedience gate]**
> `plots/stage15/obey_gate_curves.png`. Source: `runs/route_obey_v6/cont.json`.

**Composition check** — the full pipeline with a supervised decoder, run
before the unsupervised one was attempted:

```
scout pixels ──► frozen JEPA ──► logistic probe ──► route ──► frozen executor
                                 (SUPERVISED — labels)

held-out probe accuracy 1.000 · decode accuracy 1.000
success 0.996 (255/256) · hazard steps 0.00
```

The 1.000 probe is a **slab-side** logistic fitted on scout-at-post frames
during this eval — not the 0.935 free-pose visibility probe of §8.1. This
established that the pipeline is sound, and isolated the remaining
question to exactly one thing: **can reward replace the probe's
supervision?**

> **[FIGURE 8b — The composition check]**
> `plots/diagnostics/fig8b_composition_check.png`. The chain drawn as it was
> run, with the one box that is not frozen marked: scout pixels → frozen JEPA →
> **supervised logistic probe** → route bit → frozen v6 executor → navigation,
> and the measurement hung under each link it certifies (held-out probe 1.000,
> in-loop decode 1.000, obeyed 1.000, success 0.996, hazard 0.00). Below, the
> outcome given scale: success against the Stage 1.5 gate's numbers for the same
> executor handed the route directly (0.935 / 0.895, canonical spawns, a
> different episode set), and hazard steps against the ~20 a blind policy pays.
> The figure exists to make the setup for §8.6 visual — every link is at
> ceiling, so the only unearned link is the probe's labels, and race v8 replaces
> exactly that link with reward. Source:
> `runs/route_obey_v6/eval_pixels_to_route.log`.

### 8.6 Race v8 — the decisive test

Executor frozen. The only learner is a route head that sees nothing but the
message. This is a contextual bandit — precisely the exploration structure
v1–v7 lacked — and it makes the message causally load-bearing **by
construction** rather than by counterfactual.

**Design.** The executor runs deterministic mean actions, frozen. The only
trainable module is a 4,483-parameter route head: message (66) → 64 hidden →
2 logits + 1 value, logits zero-initialized so the policy starts exactly
uniform with no prior on either corridor. One categorical decision per
episode, committed at step 2, credited with the whole episode's return. The
head's mistakes are priced by the hazard — the physical fact of the world —
not by an instructor that already knows the answer. The executor still
consumes the navigator's camera through a second copy of the same frozen
PixelEncoder — the head is latent-only; the body is not.

> **[FIGURE 12 — The race v8 pipeline]**
> `plots/diagrams/fig_tier2_pipeline.png` — scout camera and navigator camera
> both enter Encoder E (shared frozen weights); only the route head is
> trained; episode return is the only gradient. Source: `rl/plot_fig_pipeline.py`.

Why step 2 rather than step 0: camera frames for a just-reset environment are
stale for the first few rendered steps; the composition eval measured a
chance-level decode off reset-time frames.

**Headline result — 5 seeds per condition, 6,000 episodes, lr 3e-3
annealed** (seeds 1–3 in `runs/race_v8/`, seeds 4–5 and the anchored-`none`
floor in `runs/race_v8b/`):

| Condition | s1 | s2 | s3 | s4 | s5 | Mean ± sd | Hazard | Obedience |
|---|---|---|---|---|---|---|---|---|
| none (floor) | 0.472 | 0.522 | 0.484 | 0.510 | 0.494 | 0.496 ± 0.020 | ≤0.64 | 1.000 |
| z_t (thesis) | 0.986 | 0.998 | 1.000 | 1.000 | 1.000 | **0.997 ± 0.006** | 0.00 | 1.000 |
| oracle (ceiling) | 0.996 | 0.990 | 0.990 | 0.984 | 0.982 | 0.988 ± 0.006 | 0.00 | 1.000 |

- **Zero seed overlap. Exact one-sided rank test at 5 vs 5: p = 0.004**
  (1/C(10,5); H1 was directional: z_t > none).
- The floor lands at 0.496. Given a content-free wire, the head can only
  learn a constant corridor preference, so it pays the hazard on half the
  episodes. Its hazard steps vary by seed only because which constant it
  settles on determines how much of the slab it clips.
- z_t numerically edges a noiseless ground-truth bit on four of five seeds.
  No equivalence test was run.
- Obedience 1.000 everywhere — the frozen controller never drifts, so what is
  being measured is the route decision.
- Zero episodes discarded by the numerical guard across all runs.

The `none` floor here is the **anchored** control: a real delivery anchor
(sender position relative to receiver) with zeroed content, so geometry is
present and only content is absent. The original v8 floor zeroed the whole
wire; the two agree to within noise (0.488–0.518 vs 0.472–0.522), which is
itself evidence the anchor carries no route information for a static scout.

Two post-mortems worth keeping, both of which cost real runs:

1. Bessel-corrected `std()` on a 1-element remainder minibatch returns NaN,
   poisoning the update. Episodes finish in bursts, so the batch is rarely an
   exact multiple of the minibatch. Three runs died before this was traced.
2. A bandit head wants lr ≈ 3e-3, not the pixel-policy 3e-4. At 3e-4 the
   oracle run's logits had moved ~0.03 after 112 Adam steps — which reads as
   "not learning" and is easily misdiagnosed as a representation failure.

> **[FIGURE 9 — Race v8 headline]** `plots/race_v8/v8_race_seed_bars.png` —
> final route-optimality by condition, per-seed dots over condition means,
> 5 seeds, anchored-`none` floor. The visual point is zero overlap. Source:
> `runs/race_v8/{z_t,oracle}*.json` + `runs/race_v8b/{none,z_t,oracle}*.json`
> (same pooling as Figure 11c; original zero-wire `none` excluded).

> **[FIGURE 10 — Race v8 learning curves]**
> `plots/race_v8/v8_race_curves.png` (+ `v8_entropy_curves.png`). 5 seeds.
> Source: the same JSON pooling as Figure 9 (`curve` field, last-500 readout
> in the JSON, not the last CSV row).

> **[FIGURE 11 — Hazard contacts]** `plots/race_v8/v8_hazard_bars.png`.
> Informed conditions sit at 0.00; the none floor clips ≤ 0.64. The m7e
> no-comms crossing (~21–25 steps, a different agent) is the text reference
> for what a blind policy pays — it is not drawn, because it would squash
> this axis. Source: same pooling as Figure 9.

### 8.7 WP1 — deterministic evaluation under corruption (formerly caveat 4)

Training-time route-optimality is a trailing on-policy average of *sampled*
decisions. `spike/eval_race_head.py` freezes everything (executor mean
actions, head argmax) and re-measures under five wire conditions: intact,
zero-content (anchor kept), zero-wire, shuffled sender (each env receives
another env's real message), and Gaussian-noise content.

| Mode | z_t heads (3 seeds) | oracle head (n=1) | Coded gate |
|---|---|---|---|
| intact | **1.000 / 1.000 / 1.000** | 1.000 | ≥ 0.95 |
| zero content | 0.496–0.527 | 0.547 | \|x − 0.5\| ≤ 0.10 |
| zero wire | 0.496–0.555 | 0.488 | \|x − 0.5\| ≤ 0.10 |
| shuffled sender | 0.445–0.520 | 0.547 | \|x − 0.5\| ≤ 0.10 |
| gaussian noise | 0.410–0.484 | 0.555 | \|x − 0.5\| ≤ 0.10 |

Executor obedience 1.000 in every mode; worst hazard 0.88 steps — a
balk-clip, never a ~20-step crossing. Seeds 4–5 were not greedily evaluated.
The coded checks pass. Two amendments relative to the original work spec,
both documented in `spike/eval_race_head.py`: the equal-weight estimator
replaced a length-biased completion-stream counter after the first run
(corrupted modes had read 0.55–0.64), and the hazard expectation was rewritten
from "crossings" to "balk-clips" after observing that this executor refuses
the slab. The ±0.10 chance band is the coded threshold, not a test of
equivalence to 0.5. The decision lives in the message content, and only
there. Artifacts: `runs/diag/eval_race_head_*.{json,log}`.

> **[FIGURE 11b — Corruption controls]**
> `plots/race_v8/v8_corruption_bars.png`. 3 z_t seeds + 1 oracle seed; 256
> episodes/mode. Source: `runs/diag/eval_race_head_*.json`.

### 8.8 WP2 — the complete condition sweep (formerly caveat 3)

All remaining conditions raced under the identical v8 protocol (frozen
executor, bandit head, 6,000 episodes), in `runs/race_v8b/`. These are
**3-seed** results; do not read them at the 5-seed confidence of the
z_t/oracle/none headline.

> **[FIGURE 2 — One channel, six message contents]**
> `plots/diagrams/fig_tier2_conditions.png` — shared 66-float wire, every
> raced row carrying its measured route-optimality (n labelled per row).
> Source: `rl/plot_fig2_conditions.py` from the same pooling as Figure 11c.

| Condition | Wire | Seeds | Route-optimality |
|---|---|---|---|
| position | scout's normalized (x,y), padded to 66 | 3 | 0.480–0.490 |
| kinematic | position + constant-velocity extrapolation | 3 | 0.460–0.496 |
| z_hat (C2) | predicted next latent P(z_t, STAY) | 3 | **0.992–1.000** |
| raw_obs | full 64×64×3 frame + anchor (12,290-d) | 3 | 0.948 / 0.956 / **0.554** |

- **The falsifier is retired on pixels.** Position and kinematic sit at the
  floor in 3/3 seeds each. A static scout's pose and motion state carry
  nothing; what it *sees* carries everything. (Both are constant wires for a
  stationary scout — the identity argument predicted this, and now it is a
  measurement, not an argument.)
- **z_hat replicates Tier 1's C2 finding under pixels**: the predicted latent
  is as decodable as the current one, as expected for a deterministic,
  information-preserving transform.
- **The compression finding, sharpened.** Tier 1 found the compact latent
  beat the raw ceiling. On pixels the raw wire is 186× wider (12,290 vs 66)
  and the result is more lopsided than a mean comparison suggests: two
  raw_obs seeds reach ~0.95 (below the latent's 0.997), and one seed's
  decision entropy collapsed to ~10⁻⁶ a third of the way in — it peaked at
  0.83, fell back to chance, and froze (787k parameters on 12,290 inputs
  commit prematurely; annealing lr cannot rescue a policy with no entropy
  left). The 66-float latent went 5/5. **Compression is not just
  radio-feasible; it is easier to recruit.**

> **[FIGURE 11c — The seven-condition sweep]**
> `plots/race_v8/v8b_sweep_bars.png` — floors (3-seed position/kinematic; 5-seed
> none), percept conditions, and ceilings in one frame, the raw_obs collapse
> annotated rather than hidden. 2/3 vs 5/5 is unequal n.

---

## 9. Interpretation, Honest Caveats, and Remaining Ablations

What the evidence supports, what it does not, and — plainly — where a
reviewer will push.

### Supported

**C1-px, perceptual transfer.** The JEPA recipe on rendered RGB clears every
probe gate, and the resulting frozen latent, broadcast to a teammate for whom
the hazard is certified invisible, supports 0.997 route-optimality at 0.00
hazard contacts. Occlusion is not asserted from an algorithm but measured in
the renderer, by the same scene builder the policy trains in.

**C3, reward-only recruitment — the headline.** Nothing supervised the
decoding. No probe target, no label, no auxiliary loss, no alignment layer,
no gradient into the encoder. The episode return is a single scalar that
names nothing; the head had to invert it. It recovered the decisive bit from
a frozen, task-agnostic representation and numerically edged a noiseless
oracle on four of five seeds — now at 5 seeds, p = 0.004 vs the floor, with
greedy evaluation under corruption controls on the first three z_t seeds plus
one oracle (§8.7).

**C2 under pixels (decodability half).** The predicted latent ẑ is as
recruitable as z_t (§8.8). The *latency* half of C2 — "prediction wins when
messages age in transit" — remains open, and belongs in Tier 1 where delivery
latency exists and episodes are ~100× cheaper.

**The falsifier, re-run on pixels.** Position and kinematic wires sit at the
floor (§8.8). Tier 1's falsifier named position-sharing explicitly; it is now
measured, not argued.

Causality here is architectural, not inferential. The route **head's** only
input is the message. The frozen executor still sees the navigator camera
through a second PixelEncoder; what the head cannot do is solve the corridor
choice from ego pixels, because it is not given them. The corruption suite
(§8.7) additionally converts that structural argument into a measured one:
destroy the content, keep everything else, and performance falls to chance.

### The caveats, in the order a reviewer will raise them

1. **The decisive fact is one bit, and the oracle ties.** Still the most
   important caveat in this document. Tier 1's C1 required regimes engineered
   so a trivial low-dimensional hand-message cannot substitute. The Tier 2
   chokepoint does not meet that bar: oracle is that hand-message, and it
   matches z_t. Tier 2 therefore confirms the pipeline and the recruitment
   mechanism; it does not independently re-establish the capability claim.
   The fix is a richer decision — variable hazard depth or a multi-corridor
   scene, so the message must convey *how far in* or *which of N*, not merely
   which side. **This is the highest-value next experiment by a wide margin
   (WP3).**
2. **The race is a contextual bandit over a frozen executor, not end-to-end
   MARL on pixels.** The decomposition is scientifically justified
   (§8.3–8.5) and the analysis that forced it is a genuine contribution. But
   Tier 2 does not reproduce Tier 1's joint-learning result on pixels; it
   isolates the representation question and answers that cleanly. A reviewer
   asking "does this still learn end-to-end?" gets an honest "not yet, and
   here is the 13σ arithmetic explaining why not."
3. ~~No position or kinematic condition was raced.~~ **Resolved** (§8.8):
   both at the floor, 3/3 seeds each.
4. ~~Training averages, not a frozen-head deterministic evaluation.~~
   **Resolved** (§8.7): greedy evaluation with shuffle, noise, and zeroing
   controls on 3 z_t seeds + 1 oracle; intact 1.000 on every evaluated seed,
   all corruptions inside the coded ±0.10 band.
5. **One scene, one map seed.** The slab side is randomized per episode, but
   the corridor geometry, the baffles, and the scout's vantage are fixed.
   Generality across occlusion structures is unproven. (Seed count is no
   longer the binding constraint: 5 seeds, p = 0.004.) A second occlusion
   topology (WP4) is the answer.
6. **The rung is sealed**, removing Tier 1's message-free escape by
   construction. The justification is measured (the continuous-control detour
   is nearly free) and Tier 1's countervailing reason does not apply here.
   But it is a simplification in the thesis's favour and belongs in the text,
   not in a code comment.
7. **The static-scout simplification carries over from Tier 1** — and is more
   load-bearing here, since the scout's fixed vantage is what makes "absence
   is signal" a stable cue. It is also why position/kinematic are constant
   wires (§8.8): the floors confirm the information account, but a mobile
   scout would make those baselines non-trivial. The mobile-scout ablation
   remains the first answer to "is this still a swarm?"

---

## 10. Risks and Mitigations (living table)

| Risk | Likelihood | Status / Mitigation |
|---|---|---|
| Occlusion does not survive extrusion | High → **materialized, resolved** | 10 hazard pixels leaked to the choice point. Staggered baffles designed against the measurement; gate passes at 0 px on both choice points, both slab placements. Gate and env share one scene builder, so this cannot silently regress. |
| JEPA recipe fails to transfer to pixels | Medium → did not materialize | Tier 1's class-weighted reconstruction fix was ported pre-emptively rather than rediscovered. All four gates passed on the first checkpoint. |
| Latent carries the info but the policy can't use it | High → **materialized, resolved, re-scoped** | Six race generations. Root cause was not the encoder — a noiseless oracle bit failed identically. Real cause: continuous-action exploration never samples the alternative route (0/128; 13σ). Resolved by the Phase 3.5 decomposition. |
| Exploration fixes distort the policy gradient | Medium → managed | AR(1) has marginal N(μ, σ) per step, so ratios stay exact when actions are re-scored under the σ they were drawn with. v7 nonetheless failed for a different reason (1/σ² attenuation) — which is why v8 abandoned noise-based coverage for categorical structure. |
| Message recruited for the wrong function | Unlisted → materialized, quantified | v7's oracle was strongly message-driven (2.69 vs 0.23) but gated advance-vs-balk inside the chosen corridor; corridor choice unchanged in 100% of rollouts. Directly motivated making route choice a first-class decision variable. |
| Separation reflects something other than decision-time message use | Low → **structurally excluded, now also measured** | The v8 head's only input is the message; no ego pathway exists. The corruption suite (§8.7) measured it anyway: intact 1.000, shuffle/noise/zero at chance. |
| Position/kinematic could match the latent | ~~Open~~ → **closed** | Raced under the v8 protocol (§8.8): both at the floor, 3/3 seeds. |
| The task is too small to need a latent | Materialized, acknowledged, **unresolved** | oracle ties z_t. Acknowledged as a scope limit rather than papered over. Resolution requires a richer decision space (WP3). |
| raw_obs ceiling embarrasses the latent | Did not materialize — **reversed** | The 12,290-d wire is *less* reliable than the 66-d latent (2/3 vs 5/5 seeds; one entropy collapse). Compression wins on optimization, not just bandwidth. |
| Toolchain instability corrupts results silently | Medium → mitigated | Driver 550 vs validated 580: cuDNN disabled at import; forced exit on post-Kit crashes so dead jobs cannot masquerade as running; NaN sanitization at the message boundary with per-episode rejection (0 rejections in the final runs). |

---

## 11. Conclusion and Next Steps

Phases 0–3 are complete, including the confirmatory program (WP0–WP2). The
instrument was built and gated against a measured occlusion criterion, the RL
core proven on raw pixels, the message validated before freezing, the
executor built and gated to obey a route command, the decisive race run to a
clean verdict, and that verdict then hardened: **a frozen self-supervised
perceptual latent, broadcast over a physics-accurate pixel pipeline, is
decodable from task reward alone, numerically edging a noiseless oracle on
4/5 seeds — 0.997 ± 0.006 (sd) against a 0.496 ± 0.020 (sd) floor, 5 seeds,
zero overlap, p = 0.004 vs the floor, zero hazard contacts — while position,
kinematics, silence, and every corrupted wire sit at chance, and the
raw-pixel firehose proves harder to learn from than the 66-float latent it
was supposed to embarrass.**

The intellectual centre of the tier is not that number. It is the diagnosis
that produced it: Tier 1's gridworld action space was silently doing the work
of temporal abstraction. Once that abstraction was removed, no amount of
tuning could recover the result, because the alternative route was a 13σ
event the optimizer literally never observed. Restoring it as architecture
recovered the finding immediately. That is a lesson about porting MARL
communication results to continuous control generally — and it is the part
most worth writing up for an outside audience.

### Remaining work, ordered by value

1. **WP3 — a decision richer than one bit.** Variable hazard depth
   (preferred) or a multi-corridor scene. This is what re-establishes Tier
   1's task constraint on pixels and turns "the pipeline works" into "the
   latent is necessary." Requires: offline signal-budget check before
   training, occlusion re-gate after any scene edit, a positive control, a
   widened route head, and a dual oracle (hand-coded bit vs hand-coded rich
   message) so the comparison is honest. Highest value by a wide margin.
2. **WP5 — the latency sweep for C2 — in Tier 1**, where uniform delivery
   latency already exists and episodes are orders of magnitude cheaper.
   Running it first on pixels would pay ~100× the compute for a weaker
   version of the same measurement.
3. **WP4 — a second occlusion topology**, to convert caveat 5 from a
   limitation into a robustness result.
4. **WP6 — mobile-scout ablation**, to retire "is this still a swarm?" and
   make the position/kinematic baselines non-trivial.

Completed and retired from this list: position/kinematic conditions (§8.8),
frozen-head greedy eval + corruption controls (§8.7), the raw_obs ceiling
(§8.8 — and it produced a second headline: compression wins on optimization
reliability, not just bandwidth).

### Phase 4 — sim-to-real: in progress (parallel track)

The bridge to the Wu Lab RoboMaster testbed is in progress and nothing about
it is claimed here. The design intent is already visible in the Tier 2
instrument, which is why several choices look over-specified for a simulator:
actions are cmd_vel matching `/robomaster_N/cmd_vel`, the camera sits at
0.2 m (confirmed to match the physical mount), the footprint is the S1's,
and the message is 64 floats — small enough to be a plain ROS2 topic at
control rate.

The target platform is the lab's retrofit RoboMaster: DJI chassis, Jetson
Orin NX 16 GB, Pi HQ camera on CSI with lab-owned OpenCV rectification
(target: the sim's 82.3° pinhole), ROS2 in Docker, Freyja LQR velocity
control. The lab has already demonstrated zero-shot sim-to-real MARL and
TensorRT inference at 20 ms on this hardware — every hard deployment problem
has a working precedent on this exact platform.

The architectural bet was that **only the encoder must cross the reality
gap**, because everything downstream would consume latents rather than pixels.
That is not the implementation. Two pixel stacks share one frozen
PixelEncoder: the scout's camera on the way onto the wire, and the
navigator's camera inside the executor. The route head is latent-only
(4,483 parameters, re-fittable in a session). The executor is not
modality-agnostic.

```
     ┌────────────── the reality gap (twice) ──────────────┐
     │                                                     │
real scout cam ──► ENCODER ──► latent ──► head ──► route bit
real nav  cam ──► ENCODER ──► z_ego  ──► executor ──► cmd_vel
                   ▲                         ▲
                   │                         │
           must transfer or           frozen, but still
           be re-fit on real          a pixel stack
           frames
```

The v8 result is evidence that the *task-coupled* module is 4,483 parameters
recruited from reward alone — re-fittable on hardware in a session. It is not
evidence that the executor is latent-only.

**The gating measurement has now been run, and the encoder does not
transfer.** The slab-side probe is fitted on sim latents and persisted
(`spike/fit_slab_probe.py` — 1.000 held-out on 512 samples); the real-frame
evaluator (`spike/eval_real_frames.py`) is Isaac-free and reports latent
health plus zero-shot probe transfer against a pre-registered decision rule
(≥0.9: the encoder transfers, deployment is plumbing; ≤0.6:
domain-randomize the JEPA data and retrain — which is the interesting
scientific result of Phase 4, not a failure). Capture protocol and tooling:
`spike/capture_real_frames.py` (ROS2, session-folder labels, rectified
frames only).

Run against the 15 Aug handoff (`handoff/analyze_handoff.py`, 13 frames, 11
labelled), pooled accuracy is 0.636, which lands nominally in the rule's
"partial" band. **That reading is wrong, and the number should not be quoted
without its baseline.** The probe predicts "hazard present" on all 13 frames,
including both empty-arena controls and both green pads: TPR 1.000, TNR
0.000. With 7 present and 4 absent, a constant predictor that ignores the
image scores the same 0.636, so balanced accuracy is 0.500 — chance. The
threshold-free ranking is *below* chance at AUC 0.286, and the best accuracy
achievable over every possible threshold is also 0.636, so recalibration
cannot recover the bit. Applying the pre-registered rule to the
chance-corrected statistic gives **no transfer**, and the ≤0.6 remedy is the
one that applies. This is the same majority-baseline correction already
applied to the hazard-visible probe in §8.1 (0.935 against a 0.857 majority);
the transfer number simply had not received it. Note also that at 7 present /
4 absent the rule's 0.6 boundary is decided by the capture set's class
balance rather than by the encoder — one additional negative frame would have
moved the same saturated probe from "partial" to "no transfer".

**The mechanism is more specific than "domain-randomize", and it implicates
the test as well as the encoder.** The probe's response is graded in the
hazard's *apparent size*, and it only separates when the slab is large: at
>25 % of the view its mean logit is +24.2, but at 2–4.5 % coverage it is
+0.66 against +0.46 for no hazard at all. The real captures put the pad at
2–21 % of the view — inside the sim coverage range, so apparent size is not
itself the gap, but mostly inside the band where **the probe has no usable
signal in simulation either**. Layered on top, the real background alone
displaces the logit by +9.5 relative to the sim no-hazard band (measured on
the pad-free controls), roughly 47× what the pad is worth at matched
coverage, and the pad at ≥2 m then moves the logit −3.4 the *wrong* way
relative to those controls. So the handoff asked the probe a question it
cannot resolve in-domain, against a nuisance shift an order of magnitude
larger than the signal. Two independent fixes are needed: the capture
protocol must present the hazard at the apparent size the encoder was trained
on (or the probe must be retrained for small-object sensitivity), *and* the
JEPA data needs appearance randomization to remove the background
displacement. Until both are done the handoff cannot be scored as a clean
test of the architectural bet.

**Two follow-up measurements (19 Aug evening) sharpen the diagnosis in the
encoder's favour.** First, the *latent* is not the small-object bottleneck:
a hazard-visibility probe fitted directly on sim latents with balanced class
weighting separates 2–8 %-coverage hazards from hazard-free frames at **AUC
0.986** on held-out streams (`spike/gate_probe_coverage.py`, gate A). The
deployed slab-side probe direction simply never used that information.
Second, the hazard bit **survives the reality gap inside the latent**:
refitting the 65-parameter probe on the real handoff latents recovers
**0.857 balanced accuracy leave-one-out from just 11 frames** (misses:
`red_100`, `red_r_280`), against 0.500 for the sim-fitted direction, while a
trivial raw-pixel red-fraction rule scores 1.000. The encoder degrades the
bit but does not destroy it — which makes the pre-registered fallback
("re-fit the probe on real frames in a session") the shortest path to a
working chain. Tooling for that session now exists:
`spike/refit_probe_real.py` (stratified-CV refit with a balanced-accuracy
≥ 0.90 gate, emits a drop-in probe checkpoint) and
`handoff/CALIBRATION_BRIEF_20260819.md` (the robot-side capture protocol:
~160 frames, ≥ 40 % negatives, distance/lateral/lighting-stratified, with a
held-out validation session). In parallel, `rl/train_jepa.py
--appearance-dr` now trains the encoder under seg-mask-guided appearance
randomization (per-sample backdrop tint/texture/photometrics with recon
targets unchanged) to attack the +9.5 background displacement at the source;
gates A/B/C of `spike/gate_probe_coverage.py` are the pre-registered
comparison between encoder candidates.

**Appearance randomization works, and the texture model is what decides it**
(19 Aug evening, all runs 10 epochs on the unchanged realcam20 dataset,
CPU-trained, all pass the Stage-A sim gates):

| encoder | sim 2–8 % AUC (gate A) | real zero-shot AUC | real refit LOO (gate C) | latent dist |
|---|---|---|---|---|
| baseline `jepa_realcam20` | 0.986 | 0.786 | 0.857 FAIL | 1.84 |
| DR v1: shared smooth texture field | 0.976 | 0.714 | 0.661 FAIL | 2.47 |
| DR v2: per-class two-band texture | 0.963 | **0.964** | **0.929 PASS** | 1.76 |

v1 (one smooth 8×8 field shared by floor and wall) made everything *worse*
than no randomization — the lab's carpet is pixel-level speckle, and an
augmentation that cannot produce it buys nothing but lost capacity. v2 gives
each backdrop class its own low- plus high-frequency field, and the
sim-fitted probe direction then *ranks* the real frames at 0.964 AUC — one
inversion in 28 pairs, the green pad at 2.8 m against the red pad
offset-left at 2.8 m. Zero-shot gate B still fails on the operating point
(all logits above the sim threshold), and a bias-only calibration from
pad-free frames is not robust because the clear-frame logits sit within 0.3
of the hardest positives; the full 65-parameter refit on the calibration set
remains the protocol. Two honesty caveats: the augmentation design was
iterated against the same 11 frames that score it, which is why the
calibration brief demands held-out validation sessions; and unguarded tint
draws can produce red-tinted walls (a hazard-hue collision), so v3 — running
as of this writing — adds a red-hue guard on backdrop tints. The champion by
gates A+C ships as the encoder candidate; the executor still consumes raw
pixels through its own encoder and is untouched by all of this.

The exported executor is also not usable on real imagery as-is: across the
same 13 frames it emits only 2 distinct actions with 97 % of action
components saturated at ±1, against 32 distinct actions over 128 sim frames.
The handoff was a static-frame study; the chain has never been run
closed-loop on hardware.

**On the sim side the executor search is exhausted, but the task is not.**
Fourteen executor variants were trained across four rounds on the realcam20
camera; the shipped checkpoint (`r2A_rescue6M`) scores 0.840 obeyed-decode,
0.504 success and 4.44 hazard steps per episode over 256 episodes, with
decode accuracy 1.000 — though that decode figure is an in-session supervised
probe refit on the same distribution, so it measures pipeline sufficiency
rather than generalisation. Within this search the checkpoint is
Pareto-optimal: across roughly 5,000 logged evaluation points in all fourteen
runs, not one exceeds both 0.840 obedience and 0.504 success. That is a
saturated *search*, not a task ceiling — roughly half of episodes still fail
with a perfectly decoded route bit.

The instability is the lever, not the compute. Nine of the fourteen runs peak
mid-run and then decline, and the round-3 and round-4 continuations peak
within their first 0.1–0.4 M steps and lose the competence they inherited.
Obedience is being bought with success: across `r2A_rescue6M`'s training,
success runs 0.243 → 0.027 → 0.045 → 0.119 → 0.260 while obedience climbs
monotonically 0.668 → 0.812. The run also ended stalled rather than
converged, with success rising +0.13/M over the final 30 % before rolling
over in the last 10 % as approximate KL collapsed from 0.050 to 0.005. The
next round should attack the objective and the trust-region schedule, not add
seeds or steps.

Round 5 is specified and ready to launch (`spike/launch_round5.sh`), one
lever per variant: the trainer now snapshots the smoothed joint
obedience×success peak (`*.pt.best` — end-of-run selection is what discarded
every previous peak), LR annealing is off and the budget is 12 M so the
recovery phase can finish; variant A keeps the r2A objective to isolate
selection+schedule, variant B replaces the global wrong-corridor
penalty/abort with a success bonus gated on obedience
(`rew_success_obedient_only` in `chokepoint/env.py`) so the goal-reaching
gradient is never suppressed. Blocked only on the machine's wedged CUDA
state (new contexts fail system-wide; a `nvidia_uvm` reload or reboot fixes
it but kills the two running jobs owned by another user).

One comparison to retire: the Stage 1.5 figure of 0.935/0.895 (`cont.pt`) is
not a valid headroom target for these runs. It predates the 20°-FOV camera
rebuild and was scored from canonical spawns, and in the realcam20 regime
canonical spawns are *harder* than randomized ones (`r2A_rescue6M`: 0.247
canonical success against 0.257 randomized). The best canonical success
logged in any realcam20 run is 0.447. `spike/eval_pixels_to_route.py` no
longer prints it as a reference.

Known gaps for deployment, unchanged: no domain randomization anywhere in
training (one dome light, flat diffuse colours, no texture variation, no
camera noise), and the physical arena must match the certified geometry in
the ratios the camera perceives (corridor width : wall height : camera
height; slab colour saturation).

Then: write-up for CoRL/RSS, with Tier 1 as the evidentiary core, Tier 2 as
transfer and reward-only recruitment, and the exploration-collapse diagnosis
as a standalone methods contribution.

---

## 12. Figure Manifest

| # | Figure | Status | Source |
|---|---|---|---|
| 1 | Tier 1 vs Tier 2 — same information structure, different substrate | `plots/fig1_substrate.png` | `rl/plot_fig1.py`: 2×3, rows = substrate, columns = viewpoint (world / corridor mouth / scout's post). Tier 1 from `envs/fov.compute_visible`; Tier 2 from 512² gate renders + `spike/render_overhead.py`, all seed 2 |
| 2 | One channel, six message contents — the conditions ladder, every row raced | `plots/diagrams/fig_tier2_conditions.png` | `rl/plot_fig2_conditions.py`: shared 66-float wire (2 anchor + 64 content), held-fixed channel properties, oracle below the rule as an off-ladder diagnostic; row markers read from `runs/race_v8b/*.json` + `runs/race_v8/{z_t,oracle}*.json` |
| 2b | Throughput and the semantic channel | **done** | `plots/diagnostics/fig2b_throughput.png` — `rl/plot_fig2b_throughput.py` from `runs/spike/fps_benchmark.log` + `spike/out/{rgb,seg}_e64_r64.npy` |
| 3 | The occlusion gate — pixel counts, and the pre-baffle leak reproduced | **done** | `plots/diagnostics/fig3_occlusion_gate.png` — `rl/plot_fig3_occlusion.py` from `runs/gate/occl_*.json` + `spike/out/occl_*` |
| 4 | M7 positive control — m7/m7b/m7c/m7e overlaid | **done** | `plots/diagnostics/fig4_m7_positive_control.png` — `rl/plot_diagnostics.py` from `runs/archive/m7_navsolo/m7*.csv` |
| 5 | JEPA training health | **done** | `plots/jepa/jepa_training.png` |
| 6 | Latent information content vs baselines | **done** | `plots/jepa/jepa_probes.png` |
| 7 | The exploration collapse — the decision, the arithmetic, the five measured settings | **done** | `plots/diagnostics/fig7_exploration_collapse.png` (parametrics: `fig7b_exploration_sweep.png`) — `rl/plot_diagnostics.py` from `runs/diag/exploration*.log` |
| 8 | Recruited but misused — v7 decay + the lie test | **done** | `plots/diagnostics/fig8_v7_recruited_misused.png` — `rl/plot_diagnostics.py` from `runs/race_v7/*.csv`, `runs/diag/route_choice_v7oracle.log` |
| 8b | The composition check — the frozen chain, its one supervised link | **done** | `plots/diagnostics/fig8b_composition_check.png` — `rl/plot_fig8b_composition.py` from `runs/route_obey_v6/eval_pixels_to_route.log` |
| 9 | Race v8 headline — per-seed dots, 5 seeds, anchored none | **done** | `plots/race_v8/v8_race_seed_bars.png` — `rl/plot_v8.py` headline pooling (`race_v8b/*.json` + `race_v8/{z_t,oracle}*.json`) |
| 10 | Race v8 learning curves + entropy | **done** | `plots/race_v8/v8_race_curves.png`, `v8_entropy_curves.png` — same 5-seed pooling |
| 11 | Hazard contacts (v8 scale; m7e ~21–25 is a text reference, not a line) | **done** | `plots/race_v8/v8_hazard_bars.png` — same pooling |
| 11b | Corruption controls (WP1) | **done** | `plots/race_v8/v8_corruption_bars.png` |
| 11c | Seven-condition sweep (WP2) | **done** | `plots/race_v8/v8b_sweep_bars.png` |
| 12 | Pipeline schematic, 4,483-param trainable surface highlighted | `plots/diagrams/fig_tier2_pipeline.png` | `rl/plot_fig_pipeline.py`: closed loop through the world, episode return drawn as the only gradient, the two absent pathways named |
| — | Stage 1.5 obedience gate | **done** | `plots/stage15/obey_gate_curves.png` |

Every figure the report cites is now rendered. Diagrams still drafted as ASCII
inline (§7.2 baffle before/after, §8.2 diagnostic chain, §10 reality gap) are
narrative aids rather than results; the baffle before/after is now also a
measurement in Figure 3, so §7.2's sketch is redundant with it.

Updated 2026-08-19: figures 2b, 3, 4, 7, 8 rendered
(`rl/plot_diagnostics.py`, `rl/plot_fig3_occlusion.py`,
`rl/plot_fig2b_throughput.py`), and two prose-only numbers turned into
committed artifacts — the throughput benchmark (`runs/spike/fps_benchmark.log`)
and the pre-baffle 10 px leak (`runs/gate/occl_s2_nobaffle.json`, via the new
`--no_baffles` diagnostic path). Headline race figures pool the 5-seed
anchored protocol rather than the superseded 3-seed zero-wire `none`.
