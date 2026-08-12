# Tier 2 Report — Roadmap, Results, and the Path to Hardware

Part II of the Tier 2 documentation (sections 7–12; sections 1–6 cover
motivation, claims, conditions, and architecture). Every number in this
document traces to a committed artifact; sources are named per figure.

Updated 2026-08-07: incorporates WP1 (corruption-controlled deterministic
evaluation) and WP2 (the complete condition sweep, 5-seed headline), which
resolve former caveats 3 and 4 and settle the raw_obs question.

---

## 7. Roadmap Part I — Building the Instrument (Phases 0–1)

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
| m7c/d | 0.55 | Hazard at −0.5/step made crossing cost 2× the success bonus, so the policy rationally refused to cross — slab-side split 1.00 / 0.00. Fix: −0.05/step, restoring Tier 1's ~20% ratio. Warm-starting across the reward change did not work: a refusing policy never samples the slab, so it never observes the new price. Retrained fresh. |
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

| Metric | Gate | Result |
|---|---|---|
| Hazard-visible probe (linear / MLP) | > majority | 0.935 / 0.942 (majority 0.857) |
| Goal-visible probe (linear / MLP) | > majority | 0.911 / 0.927 (majority 0.850) |
| Wall-count R² | ≥ 0.70 | 0.985 |
| Effective rank | ≥ 30 | 44.5 / 64 |
| Min per-dim std | > 0 | 0.924 (mean 1.032) |

Effective rank climbed monotonically from 6.9/64 at step 200 to 44.4/64 at
step 3400 — VICReg doing visible work against the collapse that bit Tier 1
immediately.

**Honest reading of the hazard probe.** 0.935 against a 0.857 majority is a
55% reduction in error, not a dramatic absolute margin. The base rate is high
because the scout sees the slab only when it is in the scout's own corridor.
The probe is a necessary check that the content exists; the race is the
arbiter of whether it is usable — and the race's answer (0.997) is
considerably stronger than the probe alone would predict.

> **[FIGURE 5 — JEPA training health]** `plots/jepa/jepa_training.png` —
> invariance loss (train vs held-out val) and effective rank over training.
> Source: `runs/archive/jepa_v1/jepa_v1.csv`.

> **[FIGURE 6 — Latent information content]** `plots/jepa/jepa_probes.png` —
> probe accuracies vs majority baselines; wall-count R² annotated. Baselines
> visually prominent: the point of the figure is clearing them. Source:
> `checkpoints/jepa_pixels.pt` (probe_metrics), log in
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
       a NOISELESS ORACLE BIT scored 0.44/0.53 vs 0.50 for silence.
    → A perfect message worth nothing is not a tuning problem.

DIAGNOSIS  ── route exploration ≠ action exploration ──────────────────────
    The alternative corridor was sampled 0 / 128 episodes at trained noise.
    Still 0 at double it.  Needs ~+1.3 sustained lateral over ~30 steps;
    σ/√30 ≈ 0.09  ⟹  a 14σ event.
    Six nulls, one cause: no batch ever CONTAINED the alternative route.

v7  ── coverage achieved, then optimized away ─────────────────────────────
    AR(1) noise (τ=30), lateral axis, first 40 steps → 0.22 coverage.
    Then: slab-bottom success 0.18–0.23 → 0.00 by iter 80,
          WHILE the boost was still near full strength.
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
| −0.60 (trained) | 0.55 | 0 | 0.00 | 1.00 | 0.68 |
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

Measured 0.22 top-corridor coverage at 0.51 success. Then the optimizer
undid it: slab-bottom success peaked near 0.22 in the first ~20 iterations and
decayed to 0.00–0.06 by iteration 80 — while the exploration boost was still
σ = 2.2, or 3.6× the policy's own learned noise. The policy found the other
corridor and learned to steer back out of it.

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
   correct route. Four unit tests assert exactly these properties — which is
   why the field could be trusted when training results were ambiguous.

**Gate: PASS.** Canonical spawns, both directions: obedience 0.960 / 0.985,
success 0.935 / 0.895. Under ±0.5 rad spawn-yaw jitter: 0.97 / 0.91
obedience, 0.915 / 0.835 success.

**Composition check** — the full pipeline with a supervised decoder, run
before the unsupervised one was attempted:

```
scout pixels ──► frozen JEPA ──► logistic probe ──► route ──► frozen executor
                                 (SUPERVISED — labels)

held-out probe accuracy 1.000 · decode accuracy 1.000
success 0.996 (255/256) · hazard steps 0.00
```

This established that the pipeline is sound, and isolated the remaining
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
not by an instructor that already knows the answer.

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

- **Zero seed overlap. Exact one-sided rank test at 5 vs 5: p = 0.004.**
- The floor lands at 0.496. Given a content-free wire, the head can only
  learn a constant corridor preference, so it pays the hazard on half the
  episodes. Its hazard steps vary by seed only because which constant it
  settles on determines how much of the slab it clips.
- z_t is statistically indistinguishable from a noiseless ground-truth bit
  and numerically edges it on four of five seeds.
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
> floor line at 0.50. The visual point is zero overlap. Source:
> `runs/race_v8/*.json`, `runs/race_v8b/*.json`.

> **[FIGURE 10 — Race v8 learning curves]**
> `plots/race_v8/v8_race_curves.png` (+ `v8_entropy_curves.png`). Source:
> `runs/race_v8/*.csv`.

> **[FIGURE 11 — Hazard contacts]** `plots/race_v8/v8_hazard_bars.png`, with
> the m7e no-comms floor (~21–25) as a reference line: both informed
> conditions sit at exactly 0.00 while a fully-trained blind policy pays the
> full price. Source: `runs/race_v8/*.json` + `runs/archive/m7_navsolo/`.

### 8.7 WP1 — deterministic evaluation under corruption (formerly caveat 4)

Training-time route-optimality is a trailing on-policy average of *sampled*
decisions. `spike/eval_race_head.py` freezes everything (executor mean
actions, head argmax) and re-measures under five wire conditions: intact,
zero-content (anchor kept), zero-wire, shuffled sender (each env receives
another env's real message), and Gaussian-noise content.

| Mode | z_t heads (3 seeds) | oracle head | Pre-registered |
|---|---|---|---|
| intact | **1.000 / 1.000 / 1.000** | 1.000 | ≈ training final |
| zero content | 0.496–0.527 | 0.547 | chance |
| zero wire | 0.496–0.555 | 0.488 | chance |
| shuffled sender | 0.445–0.520 | 0.547 | chance |
| gaussian noise | 0.410–0.484 | 0.555 | chance |

Executor obedience 1.000 in every mode; worst hazard 0.88 steps — a
balk-clip, never a ~20-step crossing. **All pre-registered checks pass.** The
decision lives in the message content, and only there.

One estimator lesson, diagnosed on the first run and worth keeping: a
completion-stream counter is length-biased. Correct decisions finish in ~206
steps, wrong ones in ~380+, so fast (correct) episodes are over-collected and
corrupted modes read 0.55–0.64. The fix — each env contributes exactly its
first K episodes, equal weight — collapsed every corrupted mode to clean
chance. Artifacts: `runs/diag/eval_race_head_*.{json,log}`.

> **[FIGURE 11b — Corruption controls]**
> `plots/race_v8/v8_corruption_bars.png`.

### 8.8 WP2 — the complete condition sweep (formerly caveat 3)

All remaining conditions raced under the identical v8 protocol (frozen
executor, bandit head, 6,000 episodes, 3 seeds each), in `runs/race_v8b/`:

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
> `plots/race_v8/v8b_sweep_bars.png` — floors, percept conditions, and
> ceilings in one frame, the raw_obs collapse annotated rather than hidden.

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
a frozen, task-agnostic representation to within noise of a noiseless oracle
— now at 5 seeds, p = 0.004, and confirmed by greedy evaluation under
corruption controls (§8.7).

**C2 under pixels (decodability half).** The predicted latent ẑ is as
recruitable as z_t (§8.8). The *latency* half of C2 — "prediction wins when
messages age in transit" — remains open, and belongs in Tier 1 where delivery
latency exists and episodes are ~100× cheaper.

**The falsifier, re-run on pixels.** Position and kinematic wires sit at the
floor (§8.8). Tier 1's falsifier named position-sharing explicitly; it is now
measured, not argued.

Causality here is architectural, not inferential. The route head's only
input is the message. Unlike Tier 1, there is no counterfactual needed to
rule out the policy solving the task from its own view — it cannot see its
own view. The corruption suite (§8.7) additionally converts the structural
argument into a measured one: destroy the content, keep everything else, and
performance falls to chance.

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
   here is the 14σ arithmetic explaining why not."
3. ~~No position or kinematic condition was raced.~~ **Resolved** (§8.8):
   both at the floor, 3/3 seeds each.
4. ~~Training averages, not a frozen-head deterministic evaluation.~~
   **Resolved** (§8.7): greedy evaluation with shuffle, noise, and zeroing
   controls; intact 1.000 on every seed, all corruptions at chance.
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
| Latent carries the info but the policy can't use it | High → **materialized, resolved, re-scoped** | Six race generations. Root cause was not the encoder — a noiseless oracle bit failed identically. Real cause: continuous-action exploration never samples the alternative route (0/128; 14σ). Resolved by the Phase 3.5 decomposition. |
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
decodable from task reward alone to within noise of a noiseless oracle —
0.997 ± 0.006 against a 0.496 ± 0.020 floor, 5 seeds, zero overlap,
p = 0.004, zero hazard contacts — while position, kinematics, silence, and
every corrupted wire sit at chance, and the raw-pixel firehose proves harder
to learn from than the 66-float latent it was supposed to embarrass.**

The intellectual centre of the tier is not that number. It is the diagnosis
that produced it: Tier 1's gridworld action space was silently doing the work
of temporal abstraction. Once that abstraction was removed, no amount of
tuning could recover the result, because the alternative route was a 14σ
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

The architectural bet Tier 2 was built to test is that **only the encoder
must cross the reality gap**, because everything downstream consumes latents
rather than pixels:

```
     ┌──────────────── the reality gap ────────────────┐
     │                                                 │
real camera ──► ENCODER ──► latent ──► head ──► executor ──► cmd_vel
                   ▲                     ▲          ▲
                   │                     │          │
           must transfer or       4.5k params,   frozen,
           be re-fit on real      re-fittable    modality-
           frames                 in a session   agnostic
```

The v8 result is the evidence for that bet: the task-coupled module is 4,483
parameters recruited from reward alone — the kind of thing you re-fit on
hardware in a session, not a training campaign.

**The gating measurement is fully tooled and waiting only on data.** The
slab-side probe is fitted on sim latents and persisted
(`spike/fit_slab_probe.py` — 1.000 held-out on 512 samples); the real-frame
evaluator (`spike/eval_real_frames.py`) is Isaac-free and reports latent
health plus zero-shot probe transfer against a pre-registered decision rule
(≥0.9: the encoder transfers, deployment is plumbing; ≤0.6:
domain-randomize the JEPA data and retrain — which is the interesting
scientific result of Phase 4, not a failure). Capture protocol and tooling:
`spike/capture_real_frames.py` (ROS2, session-folder labels, rectified
frames only).

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
| 9 | Race v8 headline — per-seed dots, floor line | **done** | `plots/race_v8/v8_race_seed_bars.png` |
| 10 | Race v8 learning curves + entropy | **done** | `plots/race_v8/v8_race_curves.png`, `v8_entropy_curves.png` |
| 11 | Hazard contacts vs the m7e no-comms floor | **done** (floor line pending) | `plots/race_v8/v8_hazard_bars.png` |
| 11b | Corruption controls (WP1) | **done** | `plots/race_v8/v8_corruption_bars.png` |
| 11c | Seven-condition sweep (WP2) | **done** | `plots/race_v8/v8b_sweep_bars.png` |
| 12 | Pipeline schematic, 4,483-param trainable surface highlighted | `plots/diagrams/fig_tier2_pipeline.png` | `rl/plot_fig_pipeline.py`: closed loop through the world, episode return drawn as the only gradient, the two absent pathways named |
| — | Stage 1.5 obedience gate | **done** | `plots/stage15/obey_gate_curves.png` |

Every figure the report cites is now rendered. Diagrams still drafted as ASCII
inline (§7.2 baffle before/after, §8.2 diagnostic chain, §11 reality gap) are
narrative aids rather than results; the baffle before/after is now also a
measurement in Figure 3, so §7.2's sketch is redundant with it.

Updated 2026-08-10: figures 2b, 3, 4, 7, 8 rendered
(`rl/plot_diagnostics.py`, `rl/plot_fig3_occlusion.py`,
`rl/plot_fig2b_throughput.py`), and two prose-only numbers turned into
committed artifacts — the throughput benchmark (`runs/spike/fps_benchmark.log`)
and the pre-baffle 10 px leak (`runs/gate/occl_s2_nobaffle.json`, via the new
`--no_baffles` diagnostic path).
