# RoboMaster S1 deployment handoff — 15 Aug 2026

Robot-side report for `runs/route_obey_v6` exported as `policy_deploy.pt`
(sha256 `3bc78f2909e0e940b45f79bfb92a091f0a5938afaa4b5fca38dae27c578aa18e`).

This is a status report, not a request for a specific change. Everything below is
measured on the physical robot in the lab.

---

## 1. The export itself is correct

`policy_deploy.pt` loads as TorchScript on CPU and reproduces the manifest's
`sanity_gray_frame` fixture exactly, to all printed digits, on both routes:

| input | route top | route bottom |
|---|---|---|
| uniform grey 0.5 | `-1.0000, +0.6430, -1.0000` | `-1.0000, -0.7177, -1.0000` |
| manifest expects | `-1.0, +0.6430329, -1.0` | `-1.0, -0.7177271, -1.0` |

The route one-hot is wired and effective: it changes `a1` sign as expected. The
node runs this fixture as a startup self-test and aborts if it drifts.

## 2. The main finding: the policy saturates on real lab imagery

`policy_probe.csv` is the exported artifact evaluated offline on the 13 real
frames in `pad_captures/`, preprocessed through the same path the robot uses.

**66 of 78 output channels (85%) sit at exactly ±1.0.**

Examples, route top:

| frame | scene | action |
|---|---|---|
| `ctrl_a` | empty arena | `-1.0000, -1.0000, +0.9532` |
| `red_200` | red pad 2 m ahead, centred | `-1.0000, -1.0000, -1.0000` |
| `red_280` | red pad 2.8 m ahead, centred | `-1.0000, -1.0000, -1.0000` |
| `red_r_280` | red pad 2.8 m, 35 cm right | `-1.0000, -1.0000, -1.0000` |
| `green_200` | green pad 2 m ahead, centred | `-0.7186, +1.0000, +1.0000` |
| `rg_split_280` | red left, green right at 2.8 m | `+0.8253, +1.0000, +1.0000` |
| uniform grey (fixture) | — | `-1.0000, +0.6430, -1.0000` |

Two things to note. Visually distinct scenes collapse to the same fully saturated
`(-1, -1, -1)`, and the yaw channel `a2` flips between its extremes for scenes that
differ only slightly. The empty-arena frames drive `a2` to `+0.95`, the opposite
extreme from the grey fixture's `-1.0`. The policy is not ignoring the image — it
responds strongly — but on this distribution it responds by pinning the action
limits rather than producing anything smooth.

We have not run the policy armed against the live camera. Given the above, doing so
would produce a constant saturated command, so we are holding until the appearance
gap is understood.

## 3. Measured camera geometry vs the sim camera

Measured from two tape-measure references laid perpendicular to the optical axis at
150 cm and 280 cm (`pad_captures/calib_tape_*`).

| quantity | measured on robot | `deploy_manifest.json` sim_camera |
|---|---|---|
| lens HFOV | 54° ± 2 (55.4° and 50.4° from the two tapes) | — |
| **HFOV after centre crop** | **32.0°** (29.7°–32.9°) | **82.3°** |
| camera height | 18 ± 1 cm | 20 cm |
| camera pitch | 2.1° below horizontal | not specified |
| source resolution | 640×360 (16:9) | 64×64 render |

Camera height agrees with the sim to within 2 cm. The horizontal field of view does
not: `tan(82.3/2) / tan(32.0/2) = 3.05`, so the real 64×64 input is roughly a **3×
magnified centre crop** of what the sim renders from the same pose. A 60 cm pad at
2.8 m fills about 46% of the real frame width.

Scale sanity check: with this calibration the known 60 cm pads measure 58–67 cm
across in all frames, so the pinhole fit is good to about 8% over the crop region.
The lens is a fisheye; the two tape distances imply focal lengths of 610 px and
680 px, and that 11% spread is the residual distortion the pinhole model does not
capture.

## 4. Robot-side preprocessing actually applied

Implemented in `code/policy_runner.py` and identically in `code/fov_check.py`:

1. `sensor_msgs/Image` via `cv_bridge` with `desired_encoding="rgb8"`
2. **rotate 180°** — the camera module is mounted inverted and the driver does not
   correct it (see `reference/BUGREPORT_camera_source_timer.md`)
3. centre square crop: 16:9 → 1:1, keeping 56.25% of width and full height
4. resize to 64×64, `cv2.INTER_AREA`
5. `float32 / 255.0`, HWC → CHW, batch dim

Action scaling and safety caps as deployed: `vx = a0 × 0.5`, `vy = a1 × 0.5`,
`wz = a2 × 1.5`, then hardware caps of ±0.15 m/s and ±0.5 rad/s. At 10 Hz.

## 5. Capture set

13 frames, one robot pose, robot stationary throughout; only the pads moved.
Each capture has `_full.jpg` (640×360 as published, rotation corrected),
`_crop.jpg` (the centre square), `_64.png` (upscaled preview of the network input),
`_64.npy` (the exact `(64,64,3)` float32 array fed to the network) and
`_geometry.json` (crop box, FOV, intended and **measured** pad placement in cm).

| tag | contents |
|---|---|
| `calib_tape_150`, `calib_tape_280` | scale references, 60 cm and 100 cm of blade |
| `ctrl_a`, `ctrl_b` | empty arena, opening and closing |
| `red_100`, `red_150`, `red_200`, `red_280` | red pad centred at 1.0/1.5/2.0/2.8 m |
| `red_l_280`, `red_r_280` | red pad at 2.8 m, 35 cm left and right |
| `green_200`, `green_280` | green pad centred at 2.0 and 2.8 m |
| `rg_split_280` | red 35 cm left and green 35 cm right, both at 2.8 m |

Pads are 60 × 60 cm interlocking foam. Every pad's measured centre offset is within
3.5 cm of its target, the left/right pair is symmetric to 2.2 cm, and `ctrl_a` vs
`ctrl_b` differ by a mean of 1.70/255 with a 7/255 peak, which bounds both robot
movement and exposure drift across the whole session. `red_100` deliberately
overflows the crop; every other pad is fully inside it.

## 6. Caveats on the bench logs

`bench_logs/` holds two `policy_runner` sessions. **Both were driven by
`fake_camera.py`, which publishes a uniform grey field, so every logged frame is
grey 128 and carries no scene information.** They are included only as evidence
that the control path works: 10 Hz timer, correct action scaling, safety caps
clamping `-0.5 → -0.15` m/s and `-1.5 → -0.5` rad/s, arm/disarm zeroing the output,
and clean zeroing on shutdown. Do not read them as behaviour on real imagery.

Known defect in our own logging: `frame_age_s` and `header_age_s` are written as
`NaN` in `steps.csv`. The watchdog itself works; the logged columns do not.

## 7. Contents

```
README.md                 this report
policy_probe.csv          exported policy evaluated on all 13 real frames + 5 references
deploy_manifest.json      the manifest the robot loaded, unmodified
pad_captures/             13 captures × 5 files
bench_logs/               2 fake_camera sessions (grey input — see section 6)
code/
  policy_runner.py        the deployed ROS 2 node
  fov_check.py            single-frame capture and geometry measurement
  capture_session.py      session sequencing and validation
  probe_policy.py         regenerates policy_probe.csv
reference/
  CAPTURE_PLAN.md         capture protocol and arena layout
  BUGREPORT_camera_source_timer.md   camera driver defects found on the Jetson
CHECKSUMS.sha256
```

Environment: ROS 2 Humble in Docker, CPU-only PyTorch, camera driver on the Jetson
at 640×360 (1080p over WiFi could not sustain the frame rate).
