# WP7 calibration brief — robot side, 19 Aug 2026

You are the agent on the commanding PC (robot LAN, ROS2, the machine that ran
the 15 Aug capture session). Your job tonight is **Stage 1: capture a
calibration + validation frame set**. Stages 2 and 3 are described so you know
where this goes, but they are gated — do not start them until the sim side
ships the artifacts listed at the end.

## Context — what the 15 Aug handoff found, and what changed

The deployed chain is: frozen JEPA pixel encoder (64×64×3 → 64-D latent) →
linear hazard probe (65 params) → 1-bit route command → executor. Your
`policy_probe.csv` finding (85 % of outputs saturated at ±1) was confirmed and
extended on the sim side:

- The sim-fitted probe says "hazard present" on **all 13** of your frames,
  including both empty-arena controls. Its pooled accuracy of 0.636 is exactly
  the class base rate (7 hazard / 4 clear) — chance-level, dressed up by class
  imbalance. All metrics are now balanced.
- The failure is the probe *direction*, not the latent space: refitting the
  probe on the real latents recovers 0.857 balanced accuracy from just 11
  frames (leave-one-out). The encoder degrades the hazard bit but does not
  destroy it.
- Root cause measured in sim: the lab backdrop (carpet, wall tint, lighting)
  displaces the probe logit ~47× more than the pad itself does at 2–2.8 m
  apparent sizes. A domain-randomized encoder (`jepa_realcam20_dr2.pt`,
  trained 19 Aug) closes most of this: its sim-fitted probe now ranks your
  13 frames at 0.964 AUC (one error: green pad at 2.8 m vs red pad
  offset-left at 2.8 m). Only the decision threshold is still wrong, which
  is exactly what your calibration set fixes. The refit below works with
  either encoder; both get re-scored on your captures.

The pre-registered fallback is exactly this: **re-fit the 65-param probe on
real frames in a session**. Your captures are that session.

## Stage 1 — the capture task (do this now)

### Ground rules

- Reuse the 15 Aug pipeline **unchanged**: `fov_check.py` /
  `capture_session.py` preprocessing — 640×360 capture → crop
  `[x=140, y=0, w=360, h=360]` → rotate 180° → resize 64×64 → float32 [0,1],
  saved as `*_64.npy` with the geometry JSON alongside. Keep the full-res
  JPG/PNG too.
- **Ignore any older instruction to rectify to the sim's 82.3° pinhole**
  (e.g. in `capture_real_frames.py`'s docstring — it is stale). The sim was
  rebuilt on 15 Aug to match your measured optics: the effective 32° HFOV of
  the central crop. No rectification beyond your standard pipeline.
- Camera geometry as measured on 14–15 Aug: mount height 0.20 m, pitch 2.1°
  down, lens 54° HFOV (measured), 32.0° effective after crop. If the mount has
  been touched since, re-run `fov_check.py --tag remount` first and note it.
- Keep the drift discipline from `capture_session.py`: control frame at
  session start and end; abort the session if drift ≥ 4 levels.
- One session folder per condition. **Folder names carry the labels**:
  `hazard_<desc>` (red pad visible) or `clear_<desc>` (no red pad). The
  sim-side refit tool parses these prefixes — nothing else is read for labels.

### Capture matrix (~160 frames, ≥40 % negatives)

The 15 Aug set was 7 hazard / 4 clear; that imbalance is what corrupted the
pre-registered metric. Do not repeat it.

**Hazard sessions (~80 frames)** — red 60×60 pad:

| distance | lateral offsets | frames each |
|---|---|---|
| 1.0 m | 0 | 3 |
| 1.5 m | −35 cm, 0, +35 cm | 3 |
| 2.0 m | −35 cm, 0, +35 cm | 3 |
| 2.5 m | −35 cm, 0, +35 cm | 3 |
| 2.8 m | −35 cm, 0, +35 cm | 3 |

For the 3 frames per placement, move the ROBOT slightly between frames
(±5 cm position, ±3° yaw) rather than the pad — viewpoint diversity is what
the probe needs. Repeat the 1.5 m and 2.8 m rows under a **second lighting
condition** (half lights, or blinds changed — whatever you can toggle;
record which in the session name, e.g. `hazard_150_dim`).

**Clear sessions (~70 frames)** — same viewpoint discipline:

- `clear_empty`: empty arena, 12 frames from varied poses (include the exact
  poses used for the hazard rows).
- `clear_green`: green pad at 1.5 / 2.0 / 2.8 m centred, 3 frames each.
- `clear_distractor`: non-red clutter a robot could plausibly see — cardboard,
  a dark jacket, a blue bin — at 1.5–2.8 m, ~12 frames. **Nothing red.**
- `clear_empty_dim`: 8 frames under the second lighting condition.

**Held-out validation (~30 frames, capture LAST)**: `hazard_val` and
`clear_val` — ~15 frames each, fresh pad placements at in-between distances
(1.25 / 1.75 / 2.4 m), a lighting state you did not use above if possible.
These are never used for fitting; they decide the gate. Keep them in
separately named folders.

### Deliverable

Tar the session folders (npy + geometry JSONs + full-res images) and send
them back the same way as the 15 Aug handoff, with a short manifest: session
name → what it contains → lighting state → anything that went wrong.

## What the sim side does with it (so you know the acceptance bar)

1. `spike/refit_probe_real.py --sessions <hazard_* clear_*> --out
   checkpoints/slab_probe_real.pt` — stratified 5-fold CV, gate:
   **balanced accuracy ≥ 0.90**, then a final check on your `*_val` sessions.
2. The same set re-scores the domain-randomized encoder candidate
   (`spike/gate_probe_coverage.py`, gates A/B/C).
3. If the gate passes, you get back: `slab_probe_real.pt` (drop-in for the
   robot node: raw latent in, `logit = z @ w + b`, hazard iff logit > 0),
   possibly a new `jepa_realcam20_dr.pt`, and an updated deploy manifest with
   fixture values.

## Stage 2 — live decode-only validation (GATED: wait for the probe)

Robot stationary, encoder+probe running onboard, **no motion commands**.
Script ≥ 100 decisions across a placement grid like Stage 1's (fresh
placements), log `(placement, ground truth, logit, decision)` per frame.
Gate: balanced accuracy ≥ 0.95 and no systematic failure at any single
distance band. Deliverable: the decision log CSV.

## Stage 3 — closed-loop (GATED: do not run)

The current executor saturates on real imagery (your own §2 finding) and is
being retrained sim-side. Do **not** drive from the policy until a new export
arrives with its own fixture and a Stage-2-style gate. `policy_runner.py`
stays parked.

## Quick reference

- Preprocessing: 640×360 → crop [140, 0, 360, 360] → rot180 → 64×64 f32
- Pad: red 60×60 cm (hazard analog); green pad and any clutter are negatives
- Labels: session folder prefix `hazard_` / `clear_` — nothing else
- Balance: ≥ 40 % negatives, every distance band represented in both classes
- Controls: drift check start/end of every session, abort ≥ 4 levels
- Send back: npy + JSON + full-res, plus the manifest
