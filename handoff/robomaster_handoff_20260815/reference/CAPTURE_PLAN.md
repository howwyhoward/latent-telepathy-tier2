# Pad capture plan (sim-to-real appearance probe)

Purpose: produce a set of real 64x64 frames that differ from each other in exactly
one controlled way, so that a difference measured by the slab probe is attributable
to the pad and not to the room, the pose, or the lighting.

The first attempt (`fov_check/archive/2026-08-14_uncontrolled_pose/`) failed this
requirement: the robot was re-aimed between captures, so background and pad both
changed together. Only `red_far` was geometrically clean. This plan fixes that.

---

## The one rule

**The robot does not move for the entire session. Only the pads move.**

Everything else in this document exists to serve that rule.

Before the first capture:

1. Park the robot facing a **blank stretch of wall**, square on (not angled), with the
   lens about **300 cm** from it. This keeps the composition that made `red_far` the
   only usable frame last time — floor plane, depth, clean horizon — while leaving room
   in front of the robot for a distance ladder. Last time the standoff was 200 cm, which
   caps pad distance at 200 cm because the pads sit between the robot and the wall.
   Do not relocate to a longer wall to gain more standoff: the extra distance buys very
   little apparent-size range, and a longer wall is far less likely to be a clean blank
   background. Holding the background fixed and featureless matters more.
2. **Tape the floor at all four wheels.** If the robot gets bumped, restore it
   before continuing, and note it.
3. Clear the other RoboMasters and any boxes out of the field of view.
4. Fix the lighting: overhead lights on, blinds unchanged, and **stand behind the
   robot** for every capture so you never cast a shadow into frame.
5. Mark the optical centreline on the floor with tape, plus cross-marks at
   100 / 150 / 200 / 250 / 280 cm along it. You will place pads against these.
   280 rather than 300 so the far pad sits just clear of the wall rather than jammed
   against the baseboard.

## Why placement is measured, not eyeballed

The policy does not see the full camera frame. Preprocessing takes a centre square
crop, keeping only the middle **56%** of the width (x 140-500 of 640), then resizes
to 64x64. Anything outside that band is invisible to the network.

That is exactly how `red_left` was lost: the pad was at the left edge of the *full
frame*, which is far outside the crop, so the policy input contained 0.1% red —
statistically identical to the no-pad control.

`fov_check.py` now checks this for you at capture time. Pass `--expect red`,
`--expect green`, or `--expect none` and it will print a `*** WARNING` if the pad
is outside the crop or clipped by its edge, and `OK` if placement is good.
**If you see a warning, fix the placement and recapture before moving on.**

## MEASURED: the field of view is 3x narrower than the simulator's

The 120 deg figure previously in the geometry JSON was copied from `camera_proc`'s
config and had never been measured. It is wrong by more than a factor of two.

Measured 2026-08-14 from a tape blade laid perpendicular to the optical axis:

| frame | blade | pixels | cm/px | full-frame HFOV | after the 0.5625 crop |
|---|---|---|---|---|---|
| `calib_tape_150` | 60 cm at 150 cm | 247 | 0.2429 | 54.8 deg | **32.5 deg** |
| `calib_tape_280` | 100 cm at 280 cm | 230 | 0.4348 | 52.8 deg | **31.2 deg** |

The two distances agree within 4%, so a pinhole model describes the crop region well
and the measurement can be trusted. `fov_check.py` now defaults to the measured
**54 deg** and reports `hfov_is_measured: true`.

**The policy's real view is ~32 deg horizontally against 82.3 deg in simulation.**
Comparing tangents, any object at a given distance appears **3.05x larger** in a real
frame than in the sim frames the policy trained on. Equivalently, the robot must stand
three times further from an obstacle to see it at the size sim taught it to expect.

This is the dominant sim-to-real gap and it is geometric, not chromatic. It outranks
pad colour, lighting and carpet texture combined. A 60 cm pad at 200 cm fills 52% of
the policy's input width in reality and would fill 17% in sim.

Three ways out, in increasing cost:

1. **Re-render the sim frames at 32 deg** for the appearance comparison. Costs nothing
   on the robot and isolates appearance from geometry, which is what the probe transfer
   is actually asking about. Do this first.
2. **Recover real FOV by re-enabling `camera_proc`'s fisheye undistortion** and cropping
   to 82.3 deg. `image_raw` is raw fisheye, so the periphery holds extra content that the
   centre crop currently throws away; how much is unknown until undistortion is on. Needs
   the 4-coefficient `front_camera_backup.yaml`, since the 5-coefficient default crashes
   OpenCV's fisheye API.
3. **Retrain with a 32 deg camera** to match the hardware. Correct, expensive, last resort.

### Consequences for pad placement

Usable lateral half-width is `d * tan(16 deg)` = `0.287 * d`:

| distance | crop half-width | centred 60 cm tile | max safe offset |
|---|---|---|---|
| 100 cm | +/- 29 cm | clipped both sides, fills frame | 0 |
| 150 cm | +/- 43 cm | fits, 70% of half-width | +/- 6 cm |
| 200 cm | +/- 57 cm | fits, 52% | +/- 18 cm |
| 280 cm | +/- 80 cm | fits, 37% | +/- 35 cm |

"Max safe offset" keeps `|offset| + 30 cm` inside 85% of the half-width. This is why
the lateral pair belongs at 280 cm: it is the only distance with real room to move a
60 cm tile sideways without clipping it.

---

## Tape calibration geometry

Reference for the two `calib_tape_*` captures in the session below. Repeat them at the
start of any session where the robot has been lifted or repositioned — a battery swap
counts, because it changes camera height and pitch even if the lens does not change.

### Where the tape goes

Coordinates are from the robot's lens: `y` straight out along the centreline, `x`
sideways, negative to the left.

```
                       WALL          <- lens is 300 cm from here
  ==========================================================
                          |
    20 cm from wall   L---+---R      TAPE #2: 100 cm of blade
                      |   |   |      L = (-50, 280)   R = (+50, 280)
                      |<-50->|
                          |
                          |
   150 cm from wall    L--+--R       TAPE #1: 60 cm of blade
                       |  |  |       L = (-30, 150)  R = (+30, 150)
                       |<30>|
                          |
                          |          centreline (floor marks from step 5)
                          |
                      [ ROBOT ]      lens at (0, 0)
```

The blade must run **parallel to the wall**, crossing the centreline at a right angle.
A tape pointing at the wall is foreshortened by perspective, so its pixel length
encodes distance and camera height rather than the lateral scale being measured here.
In the image it should look like a horizontal line crossing the frame, with **both ends
inside the frame** — a cut-off end has no known length and makes the frame useless.

To get "parallel" right without a protractor, measure from each end of the blade
straight back to the wall and make the two readings equal: 150 cm for tape #1, 20 cm
for tape #2. Weight both ends with something that is neither red nor green.

The extended length differs per distance because the narrow FOV means the frame is only
about 86-126 cm wide at 150 cm, against 160-235 cm at 280 cm. A 100 cm blade would not
fit in the nearer shot.

The `OK: pad placement is inside the centre crop` line does not validate the tape —
the checker only looks for red and green, and the blade is yellow. The tape frames are
measured by hand from the saved image.

## The capture session

Preferred way to run it — one command, works entirely offline with no chat needed:

```bash
# ===== [C] CONTAINER =====
python3 ~/deploy/capture_session.py
```

It prompts for each pad placement and waits, refuses to move on silently when a capture
fails (retry / skip / quit), and finishes by validating the whole set: per-capture pad
coverage, clipping, and the `ctrl_a` versus `ctrl_b` drift. Exits non-zero if the session
is not usable, so the verdict is computed before you leave the robot LAN.

- `--dry-run` walks the prompts without touching the camera
- `--validate-only` re-checks whatever is already on disk
- `--only red_150,green_200` re-shoots specific captures

The manual command list below does the same thing step by step, if you prefer it.

Place each pad with its **centre on the distance mark**, and its centre on the centreline
unless an offset is given. Offsets are to the pad's centre, negative to the left. This
matches how the offsets are defined and means the tile's near edge sits 30 cm nearer than
the quoted distance, which is accounted for in the warnings noted below.

Because the tile is a known 60 cm, the pixel width of its near edge back-solves to its
true distance on every frame, so modest placement slop is recoverable after the fact.

All captures stationary and unarmed, robot untouched.

| # | tag | pads | distance | offset | what it is for |
|---|---|---|---|---|---|
| 1 | `calib_tape_150` | none, tape | 150 cm | centred | Lateral scale at a known distance. |
| 2 | `calib_tape_280` | none, tape | 280 cm | centred | Second distance, separating distortion from scale and fixing camera height and pitch. |
| 3 | `ctrl_a` | none | - | - | Matched control. Every pad frame is differenced against this. |
| 4 | `red_150` | red | 150 cm | 0 | Scale series: one pad, one background, apparent size varying with distance alone. |
| 5 | `red_200` | red | 200 cm | 0 | as above |
| 6 | `red_280` | red | 280 cm | 0 | as above; roughly 2x smaller than #4 |
| 7 | `red_l_280` | red | 280 cm | -35 cm | Lateral pair: does the probe localise the slab, or merely detect it? At 280 cm there is room to offset a 60 cm tile without clipping. |
| 8 | `red_r_280` | red | 280 cm | +35 cm | Mirror of 7. An asymmetric response is a preprocessing or calibration bias, not a real effect. |
| 9 | `green_200` | green | 200 cm | 0 | Same geometry as 5, so colour is the only difference. |
| 10 | `green_280` | green | 280 cm | 0 | Crosses colour with scale rather than confounding them. |
| 11 | `rg_split_280` | green + red | 280 cm | green -35, red +35 | Both classes at once: does the probe separate them, or collapse to "coloured thing"? Leaves a ~10 cm gap between tiles. |
| 12 | `red_100` | red | 100 cm | 0 | Imminent hazard, deliberately clipped both sides. Apparent size saturates, so it is excluded from the scale series. Shot last because it is the capture most likely to disturb the robot. |
| 13 | `ctrl_b` | none | - | - | Repeat of #3 at the end. The difference between them is your **noise floor** — drift with nothing changed. Any probe effect smaller than this is not real. |

Captures 3 and 13 are what make the rest interpretable. Do not skip either.

Run the whole session in one sitting, in this order. It works outward from 150 cm and
leaves `red_100` until last on purpose: the 60 cm tile at close range sits under a metre
from the bumper, and reaching in that close is the most likely way to nudge the robot.
Putting it at the end means a late bump costs one frame instead of the whole session,
and `ctrl_b` immediately after will reveal it.

```bash
# ===== [C] CONTAINER =====
cd ~/deploy

# calibration at the current pose: also fixes camera height and pitch, which shift
# whenever the robot is lifted (battery swaps included)
python3 fov_check.py --tag calib_tape_150 --distance-cm 150 \
  --note "60cm of blade, PERPENDICULAR to axis, 30cm mark on centreline" --expect none
python3 fov_check.py --tag calib_tape_280 --distance-cm 280 \
  --note "100cm of blade, PERPENDICULAR to axis, 50cm mark on centreline" --expect none

python3 fov_check.py --tag ctrl_a --expect none --note "matched control, start of session"

python3 fov_check.py --tag red_150      --pad "red 60x60"   --distance-cm 150 --offset-cm 0   --expect red
python3 fov_check.py --tag red_200      --pad "red 60x60"   --distance-cm 200 --offset-cm 0   --expect red
python3 fov_check.py --tag red_280      --pad "red 60x60"   --distance-cm 280 --offset-cm 0   --expect red
python3 fov_check.py --tag red_l_280    --pad "red 60x60"   --distance-cm 280 --offset-cm -35 --expect red
python3 fov_check.py --tag red_r_280    --pad "red 60x60"   --distance-cm 280 --offset-cm 35  --expect red
python3 fov_check.py --tag green_200    --pad "green 60x60" --distance-cm 200 --offset-cm 0   --expect green
python3 fov_check.py --tag green_280    --pad "green 60x60" --distance-cm 280 --offset-cm 0   --expect green
python3 fov_check.py --tag rg_split_280 --pad "green+red 60x60" --distance-cm 280 \
  --note "green centre at -35cm, red centre at +35cm" --expect red
python3 fov_check.py --tag red_100      --pad "red 60x60"   --distance-cm 100 --offset-cm 0   --expect red

python3 fov_check.py --tag ctrl_b --expect none --note "matched control, end of session"
```

Expected warnings, with the pad's centre on the mark so its near edge is 30 cm nearer:

| capture | near edge | half-width there | 60 cm tile | expect |
|---|---|---|---|---|
| `red_100` | 70 cm | +/- 20 cm | clipped both sides | **warning, by design** |
| `red_150` | 120 cm | +/- 34 cm | fits with 4 cm margin | `OK`, but marginal — a warning here is not alarming |
| `red_200`, `green_200` | 170 cm | +/- 49 cm | comfortable | `OK` |
| everything at 280 | 250 cm | +/- 72 cm | comfortable even at +/- 35 offset | `OK` |

So: one certain warning on `red_100`, possibly one on `red_150`, and `OK` everywhere else.

Two caveats on the checker. Coloured light bounces off the foam onto the wall above it,
and a strong enough glow can register as pad pixels and inflate the bounding box, so
treat an unexpected edge warning as a prompt to look at the `_full.jpg` rather than as
proof. And if a capture prints the `!!!` failure banner it wrote nothing at all and
exits non-zero — re-run that tag before moving the pad.

### Validating the session

`ctrl_b` versus `ctrl_a` is the check that matters. Within a good session they agree to
about 2/255 mean absolute difference. If they differ by more than roughly 4/255 the robot
moved or the exposure drifted, and frames either side of the move are not comparable —
a real pad signal is only about 4/255 above control, so drift of that size swamps it.

## When you are done

```bash
# ===== [C] CONTAINER =====
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob('/root/deploy/fov_check/*_geometry.json')):
    d = json.load(open(f))
    pads = d.get('pad_diagnostics', {})
    print(d['tag'], {k: v['coverage_pct_of_crop'] for k, v in pads.items()}, d.get('placement', {}))
EOF
```

Every intended pad frame should show non-trivial coverage of the right colour, and
both controls should show zero. Then the set is ready to copy to the training server
for `spike/eval_real_frames.py`.
