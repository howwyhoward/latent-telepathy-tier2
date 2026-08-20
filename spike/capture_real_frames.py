"""WP7 first measurement — capture real frames for probe transfer (ROS2).

Target platform: the Prorok-lab retrofit RoboMaster — Pi HQ camera on CSI into
a Jetson Orin NX, published via ROS2 (Ubuntu + Docker onboard). There is no
DJI camera and no DJI SDK on this build.

Two requirements that make the measurement mean something:
  1. Capture through the exact preprocessing that feeds the deployed encoder.
     STALE-INSTRUCTION WARNING: this used to say "rectify to the sim's
     82.3-deg pinhole"; since the 15 Aug realcam20 rebuild the sim matches the
     MEASURED robot optics instead (54-deg lens, 32-deg effective HFOV over
     the central 360x360 crop of 640x360, rotate 180). Use the handoff's
     fov_check.py pipeline, no rectification beyond it.
  2. One invocation per session; the session folder name is the ground-truth
     label (slab_top_*, slab_bottom_*, background). Native resolution PNG,
     no resizing — preprocessing happens on the sim side.

Run inside the robot's ROS2 container (or any machine on the ROS2 network):

    python3 capture_real_frames.py --topic /robomaster_1/camera/image_rect \
        --session slab_top_light1 --frames 100

If the rectified topic does not exist yet, capture the raw topic AND commit
the calibration (K, D, target 82.3-deg P matrix) next to the frames so the
rectification can be applied offline — worse than option 1, better than
nothing.
"""

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FrameSaver(Node):
    def __init__(self, args):
        super().__init__("wp7_frame_saver")
        self.out = Path(args.out) / args.session
        self.out.mkdir(parents=True, exist_ok=True)
        self.count = len(list(self.out.glob("*.png")))
        if self.count:
            print(f"resuming: {self.count} frames already in {self.out}")
        self.target = self.count + args.frames
        self.min_dt = 1.0 / args.hz
        self.last_t = 0.0
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, args.topic, self.cb, 1)
        print(f"listening on {args.topic} -> {self.out}")

    def cb(self, msg):
        now = time.time()
        if now - self.last_t < self.min_dt or self.count >= self.target:
            return
        self.last_t = now
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = self.out / f"{self.count:05d}.png"
        cv2.imwrite(str(path), frame)
        self.count += 1
        print(f"\r{path}  ({self.count}/{self.target})", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, required=True,
                    help="rectified image topic, e.g. /robomaster_1/camera/image_rect")
    ap.add_argument("--session", type=str, required=True,
                    help="label folder, e.g. slab_top_light1")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--hz", type=float, default=2.0,
                    help="save rate; move the robot slowly between frames")
    ap.add_argument("--out", type=str, default="real_frames")
    args = ap.parse_args()

    rclpy.init()
    node = FrameSaver(args)
    try:
        while rclpy.ok() and node.count < node.target:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\ndone: {node.count} frames in {node.out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
