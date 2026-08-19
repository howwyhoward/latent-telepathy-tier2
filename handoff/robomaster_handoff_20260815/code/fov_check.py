#!/usr/bin/env python3
"""Capture one still from the camera and record the geometry the policy sees.

Run once per camera mode (e.g. 1920x1080, then 640x480) and compare the outputs
to find out whether changing resolution also changed the field of view. A mode
that scales the sensor preserves FOV; a mode that crops it narrows FOV, which
would confound the sim-to-real appearance comparison.

Usage: python3 ~/deploy/fov_check.py [--robot 1] [--tag before]
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

OUT_DIR = os.path.expanduser("~/deploy/fov_check")

# Measured 2026-08-14 from calib_tape_150 and calib_tape_280: a tape blade laid
# perpendicular to the optical axis spanned 247 px per 60 cm at 150 cm and 230 px per
# 100 cm at 280 cm, giving 54.8 and 52.8 deg. The two agree within 4%, so a pinhole
# model holds well enough across the region the centre crop keeps. This replaces the
# 120 deg figure copied from camera_proc's config, which was never measured and was
# wrong by more than a factor of two.
MEASURED_LENS_HFOV_DEG = 54.0
MEASURED_LENS_HFOV_SOURCE = (
    "measured 2026-08-14 from perpendicular tape at 150 cm and 280 cm "
    "(54.8 and 52.8 deg); pinhole-equivalent over the central crop region"
)

PAD_SIZE_CM = 60.0

EXIT_BAD_PLACEMENT = 3

# Generous, because it has to absorb both the ~8% scale error in the pinhole fit and
# however accurately a tile can be laid on a floor mark by hand. It still catches the
# failure that matters: a pad placed on the wrong mark entirely.
OFFSET_TOLERANCE_CM = 12.0


def crop_geometry(width, height):
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return {
        "source_resolution": [width, height],
        "source_aspect": round(width / height, 4),
        "crop_box_xywh": [left, top, side, side],
        # What fraction of the sensor's horizontal view survives the square crop.
        # This changes with aspect ratio alone: 0.5625 at 16:9, 0.75 at 4:3.
        "crop_fraction_of_width": round(side / width, 4),
        "crop_fraction_of_height": round(side / height, 4),
    }


def effective_hfov(lens_hfov_deg, crop_fraction):
    if not lens_hfov_deg:
        return None
    half = math.radians(lens_hfov_deg) / 2.0
    return round(math.degrees(2.0 * math.atan(crop_fraction * math.tan(half))), 2)


def colour_masks(rgb):
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return {
        "red": (r > 110) & (r - g > 40) & (r - b > 25),
        "green": (g > 90) & (g - r > 25) & (g - b > 10),
    }


def measure_pad(mask, focal_px, near_edge_cm, crop_left, crop_right):
    """Lateral offset and apparent width of the pad, in cm, measured on its near edge.

    Measured on the FULL frame, not the crop. Measuring on the crop biases a cut-off pad's
    apparent centre inward, i.e. toward whatever offset was asked for, so clipping made the
    placement check more likely to pass rather than less. On the full frame the tile is
    intact and the numbers are the tile's real geometry.

    The near edge is the widest row of the mask. It is the only part of the tile whose
    distance is known independently (pad centre distance minus half the tile), so it is
    the one row that converts from pixels to cm without guessing at perspective.
    """
    row = int(np.argmax(mask.sum(axis=1)))
    cols = np.nonzero(mask[row])[0]
    x0, x1 = int(cols.min()), int(cols.max())
    cm_per_px = near_edge_cm / focal_px
    centre_px = (x0 + x1) / 2.0
    return {
        "measured_offset_cm": round((centre_px - (crop_left + crop_right) / 2.0) * cm_per_px, 1),
        "measured_width_cm": round((x1 - x0) * cm_per_px, 1),
        # Exact rather than thresholded: the tile's own near edge either fits between the
        # crop bounds or it does not. Bounding-box contact inside the crop is too
        # trigger-happy, since coloured glow thrown onto the wall reaches the frame edge
        # while the tile itself sits well inside it.
        "apparent_size_valid": bool(x0 >= crop_left and x1 <= crop_right),
    }


def pad_diagnostics(square, expect, placement=None, eff_hfov_deg=None, allow_clip=False,
                    frame=None, crop_box=None):
    """Measure pad coverage on the cropped square, i.e. on what the policy sees.

    A pad placed outside the centre crop is invisible to the network even though
    it is plainly there in the full frame, so coverage is checked post-crop and
    edge contact is reported: a pad touching a crop edge is partly cut off and
    its apparent size no longer reflects its real distance.

    Where the intended placement is known, the pad's actual offset is measured and
    compared against it. Coverage and edge contact alone cannot catch a pad that is
    simply in the wrong place, which is how a tile 25 cm off target once passed.
    """
    report, warnings = {}, []
    placement = placement or {}
    side = square.shape[1]
    focal_px = None
    if eff_hfov_deg:
        # Identical for the crop and the full frame: cropping selects a sub-window of the
        # same sensor, it does not rescale it.
        focal_px = (side / 2.0) / math.tan(math.radians(eff_hfov_deg) / 2.0)
    distance_cm = placement.get("distance_cm")
    near_edge_cm = distance_cm - PAD_SIZE_CM / 2.0 if distance_cm else None
    can_measure = focal_px and near_edge_cm and frame is not None and crop_box is not None
    full_masks = colour_masks(frame) if frame is not None else {}

    for name, mask in colour_masks(square).items():
        pct = round(100.0 * float(mask.mean()), 2)
        entry = {"coverage_pct_of_crop": pct}
        if mask.any():
            cols = np.nonzero(mask.any(axis=0))[0]
            rows = np.nonzero(mask.any(axis=1))[0]
            entry["bbox_xyxy_in_crop"] = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
            entry["touches_left_edge"] = bool(cols.min() <= 1)
            entry["touches_right_edge"] = bool(cols.max() >= side - 2)
            if can_measure and pct >= 1.0 and full_masks[name].any():
                left, _, crop_side, _ = crop_box
                entry.update(measure_pad(full_masks[name], focal_px, near_edge_cm,
                                         left, left + crop_side - 1))
        report[name] = entry

    if expect in ("red", "green"):
        entry = report[expect]
        cut_off = entry.get("apparent_size_valid") is False
        if entry["coverage_pct_of_crop"] < 1.0:
            warnings.append(
                f"{expect} pad covers only {entry['coverage_pct_of_crop']}% of the crop -- it is "
                "outside the centre crop and effectively invisible to the policy. Move it toward "
                "the centreline and recapture."
            )
        elif cut_off and not allow_clip:
            warnings.append(
                f"{expect} pad is {entry['measured_width_cm']} cm across but runs past the edge of "
                "the centre crop, so the policy sees only part of it and its apparent size no "
                "longer encodes its distance. Move it toward the centreline and recapture."
            )
        want = placement.get("lateral_offset_cm")
        got = entry.get("measured_offset_cm")
        if want is not None and got is not None and abs(got - want) > OFFSET_TOLERANCE_CM:
            warnings.append(
                f"{expect} pad centre measures {got:+.1f} cm from the centreline but was meant to be "
                f"at {want:+.1f} cm, which is outside the {OFFSET_TOLERANCE_CM:.0f} cm tolerance. "
                "Re-measure the offset on the floor and recapture."
            )
    elif expect == "none":
        for name, entry in report.items():
            if entry["coverage_pct_of_crop"] > 0.5:
                warnings.append(
                    f"control frame still contains {entry['coverage_pct_of_crop']}% {name} pixels -- "
                    "the pads are not fully out of view."
                )
    return report, warnings


class StillGrabber(Node):
    def __init__(self, robot, tag, lens_hfov, rotate_180=True, placement=None, expect="skip",
                 allow_clip=False):
        super().__init__("fov_check")
        self.tag = tag
        self.lens_hfov = lens_hfov
        self.rotate_180 = rotate_180
        self.placement = placement or {}
        self.expect = expect
        self.allow_clip = allow_clip
        self.bridge = CvBridge()
        self.done = False
        self.warnings = []
        self.topic = f"/robomaster_{robot}/camera_0/image_raw"
        self.create_subscription(Image, self.topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"waiting for one frame on {self.topic} ...")

    def cb(self, msg):
        if self.done:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if self.rotate_180:
            # The camera module is mounted inverted and the driver does not correct it.
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        height, width = frame.shape[:2]
        geom = crop_geometry(width, height)
        geom["rotate_180"] = self.rotate_180

        left, top, side, _ = geom["crop_box_xywh"]
        square = frame[top:top + side, left:left + side]
        small = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)

        os.makedirs(OUT_DIR, exist_ok=True)
        stem = f"{self.tag}_{width}x{height}"
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_full.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_crop.jpg"), cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            os.path.join(OUT_DIR, f"{stem}_64.png"),
            cv2.cvtColor(cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR),
        )
        np.save(os.path.join(OUT_DIR, f"{stem}_64.npy"), small.astype(np.float32) / 255.0)

        geom["tag"] = self.tag
        geom["encoding"] = msg.encoding
        geom["captured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        geom["sim_reference"] = {"hfov_deg": 82.3, "renderer": "pinhole", "resolution": "64x64"}
        if self.lens_hfov:
            geom["lens_hfov_deg"] = self.lens_hfov
            geom["effective_hfov_after_crop_deg"] = effective_hfov(
                self.lens_hfov, geom["crop_fraction_of_width"]
            )
            measured = abs(self.lens_hfov - MEASURED_LENS_HFOV_DEG) < 1e-6
            geom["hfov_is_measured"] = measured
            geom["lens_hfov_source"] = MEASURED_LENS_HFOV_SOURCE if measured else "user override"
            geom["hfov_vs_sim_magnification"] = round(
                math.tan(math.radians(geom["sim_reference"]["hfov_deg"] / 2.0))
                / math.tan(math.radians(geom["effective_hfov_after_crop_deg"] / 2.0)), 2
            )
        if self.placement:
            geom["placement"] = self.placement

        report, warnings = pad_diagnostics(
            square, self.expect, placement=self.placement,
            eff_hfov_deg=geom.get("effective_hfov_after_crop_deg"), allow_clip=self.allow_clip,
            frame=frame, crop_box=geom["crop_box_xywh"],
        )
        geom["pad_diagnostics"] = report

        with open(os.path.join(OUT_DIR, f"{stem}_geometry.json"), "w") as handle:
            json.dump(geom, handle, indent=2)

        print(json.dumps(geom, indent=2))
        print(f"\nwrote {OUT_DIR}/{stem}_*.{{jpg,png,npy,json}}")
        for line in warnings:
            print(f"\n*** WARNING: {line}")
        if not warnings and self.expect != "skip":
            print("\nOK: pad placement is inside the centre crop.")
        self.warnings = warnings
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=int, default=1)
    parser.add_argument("--tag", default="still")
    parser.add_argument("--lens-hfov", type=float, default=MEASURED_LENS_HFOV_DEG,
                        help=f"lens HFOV in degrees (default {MEASURED_LENS_HFOV_DEG}, measured; "
                             "used to report effective post-crop HFOV)")
    parser.add_argument("--no-rotate", action="store_true",
                        help="save the frame exactly as published, without correcting the inverted mount")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--distance-cm", type=float, default=None,
                        help="camera lens to pad centre, along the optical axis")
    parser.add_argument("--offset-cm", type=float, default=None,
                        help="pad centre lateral offset from the optical axis, negative = left")
    parser.add_argument("--pad", default=None, help="pad colour and size, e.g. 'red 60x60'")
    parser.add_argument("--note", default=None, help="free-text note stored with the frame")
    parser.add_argument("--expect", choices=["red", "green", "none", "skip"], default="skip",
                        help="which pad should be visible; enables the post-crop placement check")
    parser.add_argument("--allow-clip", action="store_true",
                        help="the pad is meant to overflow the crop, as at very close range")
    args = parser.parse_args()

    placement = {k: v for k, v in (
        ("distance_cm", args.distance_cm),
        ("lateral_offset_cm", args.offset_cm),
        ("pad", args.pad),
        ("note", args.note),
    ) if v is not None}

    rclpy.init()
    node = StillGrabber(args.robot, args.tag, args.lens_hfov, rotate_180=not args.no_rotate,
                        placement=placement, expect=args.expect, allow_clip=args.allow_clip)
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and not node.done and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    publishers = node.count_publishers(node.topic)
    warnings = node.warnings
    node.destroy_node()
    rclpy.shutdown()

    if not node.done:
        # A silent no-op here previously cost six captures: the host's WiFi had flipped off
        # the robot LAN mid-session, so every run timed out, printed one line and exited 0,
        # which is indistinguishable from success in a scrollback.
        print("\n" + "!" * 72)
        print(f"!!! CAPTURE FAILED: no frame within {args.timeout}s. NOTHING was written for "
              f"tag '{args.tag}'.")
        if publishers == 0:
            print("!!! No publisher on the topic at all -- the host is probably not on the robot")
            print("!!! LAN. Check with: nmcli -t -f ACTIVE,SSID dev wifi | grep '^yes'")
        else:
            print(f"!!! {publishers} publisher(s) listed but no frame arrived. Either the camera")
            print("!!! driver is stalled, OR discovery is stale because the link dropped after the")
            print("!!! publisher was found -- a mid-capture WiFi flip looks exactly like this.")
            print("!!! Check the link first: iwgetid && ping -c2 192.168.0.175")
            print("!!! Then, if the link is fine:")
            print("!!!   ssh -t nvidia@192.168.0.175 'sudo systemctl restart camera_stream_0'")
        print("!" * 72)
        sys.exit(1)

    if warnings:
        # Distinct from 1: the frame was captured and written, it is the placement that is
        # wrong. The caller can offer to move the pad and shoot again rather than treating
        # this as a lost frame.
        sys.exit(EXIT_BAD_PLACEMENT)


if __name__ == "__main__":
    main()
