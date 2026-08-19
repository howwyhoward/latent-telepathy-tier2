#!/usr/bin/env python3
"""Run a full pad-capture session offline, prompting between captures.

Written so the whole session can be completed while the host is on the robot LAN with
no internet and no chat available. Each capture is delegated to fov_check.py, so there
is one implementation of the preprocessing and the placement check; this script only
sequences them, refuses to silently skip a failure, and validates the set at the end.

Usage: python3 ~/deploy/capture_session.py [--robot 1] [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOV_CHECK = os.path.join(HERE, "fov_check.py")
OUT_DIR = os.path.join(HERE, "fov_check")

# Drift between the opening and closing control frames, in 8-bit levels. A real pad
# signal measures about 4 levels above control, so drift at or above that swamps it.
DRIFT_GOOD = 2.0
DRIFT_FAIL = 4.0

# Minimum coverage of the cropped square for a pad that is supposed to be visible.
MIN_PAD_COVERAGE_PCT = 1.0

try:
    from fov_check import EXIT_BAD_PLACEMENT, OFFSET_TOLERANCE_CM
except ImportError:  # --validate-only outside the ROS container, where rclpy is absent
    EXIT_BAD_PLACEMENT, OFFSET_TOLERANCE_CM = 3, 12.0

SESSION = [
    dict(tag="calib_tape_150", expect="none", args=["--distance-cm", "150",
         "--note", "60cm of blade, PERPENDICULAR to axis, 30cm mark on centreline"],
         instruction="""TAPE #1 -- scale reference at 150 cm
  Extend exactly 60 cm of blade. Lay it FLAT on the floor at the 150 cm mark,
  running LEFT-TO-RIGHT parallel to the wall, with the 30 cm graduation on the
  centreline. Case ends up 30 cm left, tip 30 cm right.
  Check parallel: both ends should measure 150 cm back to the wall.
  Both ends must be inside the frame. Weight them with something not red or green."""),

    dict(tag="calib_tape_280", expect="none", args=["--distance-cm", "280",
         "--note", "100cm of blade, PERPENDICULAR to axis, 50cm mark on centreline"],
         instruction="""TAPE #2 -- scale reference at 280 cm
  Extend exactly 100 cm of blade. Same idea at the 280 cm mark, 50 cm graduation
  on the centreline. Case 50 cm left, tip 50 cm right.
  Check parallel: both ends should measure 20 cm back to the wall.
  Flip the blade so its wide face is up -- it reads far more cleanly."""),

    dict(tag="ctrl_a", expect="none", args=["--note", "matched control, start of session"],
         instruction="""CONTROL (opening)
  Remove EVERYTHING from the arena: tape, pads, your feet, the other robots.
  This frame is the reference that every pad frame is compared against."""),

    dict(tag="red_150", expect="red", args=["--pad", "red 60x60", "--distance-cm", "150",
         "--offset-cm", "0"],
         instruction="""RED at 150 cm, centred
  Pad CENTRE on the 150 cm mark, centred left-to-right on the centreline.
  This one is marginal for clipping -- an edge warning here is not alarming."""),

    dict(tag="red_200", expect="red", args=["--pad", "red 60x60", "--distance-cm", "200",
         "--offset-cm", "0"],
         instruction="""RED at 200 cm, centred
  Move the same pad back to the 200 cm mark. Do not rotate it or swap tiles."""),

    dict(tag="red_280", expect="red", args=["--pad", "red 60x60", "--distance-cm", "280",
         "--offset-cm", "0"],
         instruction="""RED at 280 cm, centred
  Same pad back to the 280 cm mark, just clear of the baseboard."""),

    dict(tag="red_l_280", expect="red", args=["--pad", "red 60x60", "--distance-cm", "280",
         "--offset-cm", "-35"],
         instruction="""RED at 280 cm, offset 35 cm LEFT
  Keep the pad at 280 cm, slide its CENTRE 35 cm to the LEFT of the centreline."""),

    dict(tag="red_r_280", expect="red", args=["--pad", "red 60x60", "--distance-cm", "280",
         "--offset-cm", "35"],
         instruction="""RED at 280 cm, offset 35 cm RIGHT
  Mirror of the last one: same distance, centre 35 cm RIGHT of the centreline.
  Try to make the offset match the left one as closely as you can -- this pair
  only means something if it is symmetric."""),

    dict(tag="green_200", expect="green", args=["--pad", "green 60x60", "--distance-cm", "200",
         "--offset-cm", "0"],
         instruction="""GREEN at 200 cm, centred
  Swap the red tile for a green one, centred on the 200 cm mark. Same geometry as
  red_200 so that colour is the only thing that changed."""),

    dict(tag="green_280", expect="green", args=["--pad", "green 60x60", "--distance-cm", "280",
         "--offset-cm", "0"],
         instruction="""GREEN at 280 cm, centred
  Green tile back to the 280 cm mark."""),

    dict(tag="rg_split_280", expect="red", args=["--pad", "green+red 60x60",
         "--distance-cm", "280", "--offset-cm", "-35",
         "--note", "red centre at -35cm, green centre at +35cm"],
         instruction="""RED + GREEN together at 280 cm
    Both tiles at 280 cm: RED centre 35 cm LEFT, GREEN centre 35 cm RIGHT.
    That leaves roughly a 10 cm gap between them."""),

    dict(tag="red_100", expect="red", allow_clip=True,
         args=["--pad", "red 60x60", "--distance-cm", "100", "--offset-cm", "0", "--allow-clip"],
         instruction="""RED at 100 cm, centred  <-- CAREFUL, THIS IS CLOSE TO THE ROBOT
  Remove the green tile. Red pad centre on the 100 cm mark.
  Its near edge lands about 70 cm from the lens, so reach in slowly and do NOT
  touch the robot. An edge warning here is EXPECTED and correct.
  This is shot last precisely because it is the capture most likely to nudge
  the robot -- if that happens, only this frame is affected."""),

    dict(tag="ctrl_b", expect="none", args=["--note", "matched control, end of session"],
         instruction="""CONTROL (closing)
  Remove everything again, exactly as for the opening control.
  Comparing this against ctrl_a is what proves the session is internally
  consistent, so do not skip it."""),
]


def banner(text, char="="):
    print("\n" + char * 78)
    for line in text.splitlines():
        print(line)
    print(char * 78)


def ask(prompt, choices):
    while True:
        got = input(f"{prompt} [{'/'.join(choices)}]: ").strip().lower()
        if got in choices:
            return got
        for c in choices:
            if c.startswith(got) and got:
                return c
        print(f"  please answer one of: {', '.join(choices)}")


def run_capture(item, robot, dry_run):
    cmd = [sys.executable, FOV_CHECK, "--robot", str(robot), "--tag", item["tag"],
           "--expect", item["expect"]] + item["args"]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0
    return subprocess.call(cmd)


def geometry_path(tag):
    for name in os.listdir(OUT_DIR):
        if name.startswith(tag + "_") and name.endswith("_geometry.json"):
            return os.path.join(OUT_DIR, name)
    return None


def npy_path(tag):
    for name in os.listdir(OUT_DIR):
        if name.startswith(tag + "_") and name.endswith("_64.npy"):
            return os.path.join(OUT_DIR, name)
    return None


def validate():
    banner("SESSION VALIDATION", "=")
    problems, notes = [], []

    missing = [i["tag"] for i in SESSION if geometry_path(i["tag"]) is None]
    if missing:
        problems.append(f"missing captures: {', '.join(missing)}")

    offsets = {}
    print(f"{'tag':16s} {'red%':>7s} {'green%':>7s} {'offset':>8s} {'want':>6s}  status")
    for item in SESSION:
        tag, expect = item["tag"], item["expect"]
        path = geometry_path(tag)
        if path is None:
            print(f"{tag:16s} {'-':>7s} {'-':>7s} {'-':>8s} {'-':>6s}  MISSING")
            continue
        d = json.load(open(path))
        pad = d.get("pad_diagnostics", {})
        red = pad.get("red", {}).get("coverage_pct_of_crop", 0.0)
        green = pad.get("green", {}).get("coverage_pct_of_crop", 0.0)
        clipped = any(pad.get(c, {}).get("apparent_size_valid") is False for c in ("red", "green"))
        got = pad.get(expect, {}).get("measured_offset_cm") if expect in ("red", "green") else None
        want = d.get("placement", {}).get("lateral_offset_cm")
        if got is not None:
            offsets[tag] = got

        status = "ok"
        if expect in ("red", "green"):
            cov = red if expect == "red" else green
            if cov < MIN_PAD_COVERAGE_PCT:
                status = "FAIL: pad not in crop"
                problems.append(f"{tag}: {expect} covers only {cov}% of the crop")
            elif want is not None and got is not None and abs(got - want) > OFFSET_TOLERANCE_CM:
                status = "FAIL: wrong place"
                problems.append(f"{tag}: {expect} pad centre is at {got:+.1f} cm but was meant to be "
                                f"at {want:+.1f} cm -- recapture this one")
            elif clipped and not item.get("allow_clip"):
                status = "FAIL: clipped"
                problems.append(f"{tag}: pad runs off the edge of the crop, so its apparent size no "
                                "longer encodes distance -- move it inward and recapture")
            elif clipped:
                status = "ok (clipped by design)"
        else:
            if max(red, green) > 0.5:
                status = "FAIL: pads still visible"
                problems.append(f"{tag}: control frame contains {max(red, green)}% pad pixels")
        got_s = f"{got:+.1f}" if got is not None else "-"
        want_s = f"{want:+.0f}" if want is not None else "-"
        print(f"{tag:16s} {red:7.2f} {green:7.2f} {got_s:>8s} {want_s:>6s}  {status}")

    # The left/right pair is only interpretable as a mirror image. Each side can sit inside
    # tolerance while the pair is still lopsided, so check the pair itself.
    left, right = offsets.get("red_l_280"), offsets.get("red_r_280")
    if left is not None and right is not None:
        asym = abs(abs(left) - abs(right))
        print(f"\nmirror pair red_l_280 / red_r_280: {left:+.1f} cm vs {right:+.1f} cm "
              f"(asymmetry {asym:.1f} cm)")
        if asym > OFFSET_TOLERANCE_CM:
            problems.append(f"the left/right pair is asymmetric by {asym:.1f} cm, so it cannot be "
                            "used to test left/right response -- recapture the offending side")
        else:
            print("  -> symmetric enough to compare the two sides")

    a, b = npy_path("ctrl_a"), npy_path("ctrl_b")
    if a and b:
        fa, fb = np.load(a), np.load(b)
        drift = float(np.abs(fa - fb).mean()) * 255.0
        peak = float(np.abs(fa - fb).max()) * 255.0
        print(f"\ncontrol drift ctrl_a vs ctrl_b: mean {drift:.2f}/255, peak {peak:.1f}/255")
        if drift < DRIFT_GOOD:
            print("  -> excellent, the robot did not move and exposure held steady")
        elif drift < DRIFT_FAIL:
            print("  -> acceptable, but there is some drift; note it when analysing")
        else:
            problems.append(f"control drift {drift:.2f}/255 exceeds {DRIFT_FAIL} -- the robot "
                            "moved or exposure shifted, so frames either side are not comparable")
    else:
        problems.append("cannot measure control drift: ctrl_a or ctrl_b is missing")

    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")
    if problems:
        banner("SESSION NOT READY\n\n" + "\n".join(f"  - {p}" for p in problems), "!")
        return 1
    banner("SESSION COMPLETE AND VALID -- ready to copy to the training server", "=")
    return 0


def hand_to_host_user():
    """Match ownership of the output dir to whoever owns ~/deploy on the host mount."""
    try:
        target = os.stat(HERE)
        for root, dirs, files in os.walk(OUT_DIR):
            for name in dirs + files:
                os.chown(os.path.join(root, name), target.st_uid, target.st_gid)
        os.chown(OUT_DIR, target.st_uid, target.st_gid)
    except (OSError, PermissionError):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and the prompts without touching the camera")
    parser.add_argument("--only", default=None,
                        help="comma-separated tags to run instead of the whole session")
    parser.add_argument("--validate-only", action="store_true",
                        help="skip capturing and just check what is already on disk")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.validate_only:
        sys.exit(validate())

    todo = SESSION
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        todo = [i for i in SESSION if i["tag"] in wanted]
        unknown = wanted - {i["tag"] for i in SESSION}
        if unknown:
            print(f"unknown tags: {', '.join(sorted(unknown))}")
            sys.exit(2)

    existing = [i["tag"] for i in todo if geometry_path(i["tag"])]
    if existing:
        banner("These tags already exist and will be OVERWRITTEN:\n  " + ", ".join(existing), "!")
        if ask("continue", ["yes", "no"]) == "no":
            print("nothing done. Move the old files aside first if you want to keep them.")
            sys.exit(1)

    banner("""PAD CAPTURE SESSION

  The robot must NOT move for the whole session. Only the pads move.
  Stand BEHIND the robot for every capture so you never cast a shadow in.
  Stay on the robot LAN throughout -- switching networks mid-capture is what
  killed the previous two sessions.

  At each step: place the pad as described, then press Enter.
  Enter = capture, s = skip this one, q = stop the session.""")
    if input("\nPress Enter when you are ready to start (or q to quit): ").strip().lower() == "q":
        sys.exit(1)

    done, skipped = [], []
    started = time.time()
    for n, item in enumerate(todo, 1):
        banner(f"[{n}/{len(todo)}]  {item['tag']}\n\n{item['instruction']}")
        got = input("Enter to capture / s to skip / q to quit: ").strip().lower()
        if got == "q":
            print("stopping early.")
            break
        if got == "s":
            skipped.append(item["tag"])
            continue

        while True:
            rc = run_capture(item, args.robot, args.dry_run)
            if rc == 0:
                done.append(item["tag"])
                break
            if rc == EXIT_BAD_PLACEMENT:
                print("\nThe frame was captured, but the pad is NOT where it was meant to be.")
                print("Read the warning above, move the pad on the floor, then retry.")
            else:
                print("\nThat capture FAILED and wrote nothing.")
                print("If the message mentioned no publisher, check: iwgetid && ping -c2 192.168.0.175")
            what = ask("retry, skip, or quit", ["retry", "skip", "quit"])
            if what == "retry":
                continue
            if what == "skip":
                skipped.append(item["tag"])
                break
            print("stopping early.")
            hand_to_host_user()
            sys.exit(1)

    mins = (time.time() - started) / 60.0
    print(f"\ncaptured {len(done)} in {mins:.1f} min" + (f", skipped {len(skipped)}" if skipped else ""))
    hand_to_host_user()
    if not args.dry_run:
        sys.exit(validate())


if __name__ == "__main__":
    main()
