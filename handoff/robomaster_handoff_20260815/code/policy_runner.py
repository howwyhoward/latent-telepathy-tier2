#!/usr/bin/env python3
"""Deployment plumbing node: runs an exported TorchScript navigation policy on a
RoboMaster S1 and logs the exact tensors it fed to the network.

The policy was trained in a specific sim scene, so behaviour in an arbitrary room
is expected to be out-of-distribution. This node exists to prove the export path
and to capture 64x64 frames for a sim-to-real appearance comparison.
"""

import collections
import csv
import hashlib
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

ROUTE_ONE_HOT = {"top": [1.0, 0.0], "bottom": [0.0, 1.0]}
SELF_TEST_TOL = 1e-3
ZERO_PUBLISH_REPEATS = 5
WRITER_QUEUE_MAX = 200


def read_manifest(path):
    if not os.path.isfile(path):
        raise SystemExit(f"FATAL: deploy manifest missing at {path}; refusing to run blind")
    with open(path) as handle:
        return json.load(handle)


def fixture_from_manifest(manifest):
    """Extract the gray-frame expectations, tolerating either manifest schema.

    The training-side export uses `sanity_gray_frame.route_{top,bottom}`; the
    deployment-side reconstruction used `self_test.expected.{top,bottom}`.
    """
    gray = manifest.get("sanity_gray_frame")
    if isinstance(gray, dict):
        found = {}
        for route in ROUTE_ONE_HOT:
            for key in (f"route_{route}", route):
                if key in gray:
                    found[route] = [float(v) for v in gray[key]]
                    break
        if len(found) == len(ROUTE_ONE_HOT):
            return found, float(manifest.get("sanity_tolerance", SELF_TEST_TOL))

    self_test = manifest.get("self_test")
    if isinstance(self_test, dict) and isinstance(self_test.get("expected"), dict):
        expected = self_test["expected"]
        if all(route in expected for route in ROUTE_ONE_HOT):
            return (
                {route: [float(v) for v in expected[route]] for route in ROUTE_ONE_HOT},
                float(self_test.get("tolerance", SELF_TEST_TOL)),
            )

    raise SystemExit(
        "FATAL: manifest has no usable gray-frame fixture "
        "(expected 'sanity_gray_frame.route_top/route_bottom' or 'self_test.expected.top/bottom')"
    )


class PolicyRunner(Node):
    def __init__(self):
        super().__init__("policy_runner")

        deploy_dir = os.path.expanduser("~/deploy")
        self.declare_parameter("robot_id", 1)
        self.declare_parameter("route", "top")
        self.declare_parameter("model_path", os.path.join(deploy_dir, "policy_deploy.pt"))
        self.declare_parameter("manifest_path", os.path.join(deploy_dir, "deploy_manifest.json"))
        self.declare_parameter("log_root", os.path.join(deploy_dir, "logs"))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("frame_timeout_s", 0.3)
        self.declare_parameter("scale_linear", 0.5)
        self.declare_parameter("scale_angular", 1.5)
        self.declare_parameter("cap_linear", 0.15)
        self.declare_parameter("cap_angular", 0.5)
        self.declare_parameter("jpeg_period_s", 2.0)
        # Robot and host clocks are not synchronised on this testbed (observed
        # ~9 month offset), so header stamps cannot drive the watchdog.
        self.declare_parameter("use_header_stamp", False)
        self.declare_parameter("start_armed", False)
        # The S1's camera module is mounted inverted and the driver exposes no
        # flip-method parameter, so image_raw arrives rotated 180 degrees. Sim
        # trained upright, so serving the raw frame is maximally out of
        # distribution. Set false only once the driver corrects it.
        self.declare_parameter("rotate_180", True)

        self.robot_id = int(self.get_parameter("robot_id").value)
        self.route_name = str(self.get_parameter("route").value).strip().lower()
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.frame_timeout_s = float(self.get_parameter("frame_timeout_s").value)
        self.scale_linear = float(self.get_parameter("scale_linear").value)
        self.scale_angular = float(self.get_parameter("scale_angular").value)
        self.cap_linear = float(self.get_parameter("cap_linear").value)
        self.cap_angular = float(self.get_parameter("cap_angular").value)
        self.jpeg_period_s = float(self.get_parameter("jpeg_period_s").value)
        self.use_header_stamp = bool(self.get_parameter("use_header_stamp").value)
        self.rotate_180 = bool(self.get_parameter("rotate_180").value)

        if self.route_name not in ROUTE_ONE_HOT:
            raise SystemExit(f"route must be one of {list(ROUTE_ONE_HOT)}, got '{self.route_name}'")

        self.manifest = self._load_manifest(str(self.get_parameter("manifest_path").value))
        self.model = self._load_model(str(self.get_parameter("model_path").value))
        self.route_tensor = torch.tensor([ROUTE_ONE_HOT[self.route_name]], dtype=torch.float32)
        self._run_self_test()

        self.armed = bool(self.get_parameter("start_armed").value)
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_rx_time = None
        self._latest_header_stamp = None
        self._step = 0
        self._last_jpeg_time = 0.0
        self._shutting_down = False
        self._camera_geometry = None
        self._status_counts = collections.Counter()
        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._ended_at = None

        self.session_dir, self.frames_dir = self._make_session_dir(str(self.get_parameter("log_root").value))
        self._csv_file = open(os.path.join(self.session_dir, "steps.csv"), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp", "step", "frame_age_s", "header_age_s",
            "raw_a0", "raw_a1", "raw_a2",
            "scaled_x", "scaled_y", "scaled_wz",
            "clamped_x", "clamped_y", "clamped_wz",
            "armed", "route", "status",
        ])
        self._csv_file.flush()

        # Disk writes must never delay the control loop or the image callback.
        self._write_queue = queue.Queue(maxsize=WRITER_QUEUE_MAX)
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

        image_topic = f"/robomaster_{self.robot_id}/camera_0/image_raw"
        cmd_topic = f"/robomaster_{self.robot_id}/cmd_vel"
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        # Separate groups so a slow 1080p conversion cannot starve the 10 Hz timer.
        self.create_subscription(
            Image, image_topic, self._image_cb, qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.arm_srv = self.create_service(
            SetBool, "~/arm", self._arm_cb, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.timer = self.create_timer(
            1.0 / self.rate_hz, self._control_step, callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.get_logger().info(
            f"route={self.route_name} robot={self.robot_id} rate={self.rate_hz:.1f}Hz "
            f"caps=+/-{self.cap_linear} m/s, +/-{self.cap_angular} rad/s"
        )
        self.get_logger().info(f"sub {image_topic} -> pub {cmd_topic}")
        if self.rotate_180:
            self.get_logger().warn(
                "rotate_180=true: correcting the inverted camera mount before the crop. "
                "Set false if the driver gains a flip parameter."
            )
        else:
            self.get_logger().error(
                "rotate_180=false: frames are served as published. If the camera mount is "
                "still inverted, the policy sees an upside-down world."
            )
        self.get_logger().info(f"logging to {self.session_dir}")
        self.get_logger().warn(
            "DISARMED: publishing zero Twist. Arm with: "
            f"ros2 service call /policy_runner/arm std_srvs/srv/SetBool \"{{data: true}}\""
            if not self.armed else "ARMED at startup"
        )

    # ---------- startup ----------

    def _load_manifest(self, path):
        manifest = read_manifest(path)
        # Raises if the fixture is unusable, so a bad manifest fails before the wheels can turn.
        self._expected_actions, self._self_test_tol = fixture_from_manifest(manifest)
        self.get_logger().info(f"manifest loaded: {path}")
        for key in ("artifact", "source_executor", "source_jepa"):
            if key in manifest:
                self.get_logger().info(f"  {key}: {manifest[key]}")
        if "action_scaling" in manifest:
            self.get_logger().info(f"  manifest action_scaling: {manifest['action_scaling']}")
            self.get_logger().info(
                f"  node scaling: linear={self.scale_linear} m/s, angular={self.scale_angular} rad/s"
            )
        if "sim_camera" in manifest:
            # Training used a narrow pinhole camera; this robot has a wide fisheye
            # lens, so the observation distribution differs before any lighting or
            # texture gap is considered.
            self.get_logger().warn(f"  sim camera was {manifest['sim_camera']} -- real lens is wide fisheye")
        manifest_rate = manifest.get("control_rate_hz")
        if manifest_rate is not None and abs(float(manifest_rate) - self.rate_hz) > 1e-6:
            self.get_logger().warn(
                f"  control rate mismatch: manifest says {manifest_rate} Hz, node running {self.rate_hz} Hz"
            )
        return manifest

    def _load_model(self, path):
        if not os.path.isfile(path):
            raise SystemExit(f"FATAL: policy missing at {path}")
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        with open(path, "rb") as handle:
            self.model_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.get_logger().info(f"policy loaded: {path} (sha256 {self.model_sha256[:12]})")
        return model

    def _run_self_test(self):
        expected = self._expected_actions
        tol = self._self_test_tol
        gray = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
        for name, one_hot in ROUTE_ONE_HOT.items():
            with torch.no_grad():
                action = self.model(gray, torch.tensor([one_hot], dtype=torch.float32))
            if tuple(action.shape) != (1, 3):
                raise SystemExit(f"FATAL: self-test route={name} shape {tuple(action.shape)} != (1, 3)")
            got = [float(v) for v in action.squeeze(0)]
            max_err = max(abs(g - e) for g, e in zip(got, expected[name]))
            if max_err >= tol:
                raise SystemExit(
                    f"FATAL: self-test route={name} mismatch: got {got}, expected {expected[name]}, "
                    f"max_err={max_err:.3e} >= {tol}"
                )
            self.get_logger().info(
                f"self-test route={name}: {[round(v, 4) for v in got]} (max_err={max_err:.2e}) OK"
            )
        self.get_logger().info("SELF-TEST PASSED: model load and tensor plumbing verified")

    def _make_session_dir(self, log_root):
        session = os.path.join(log_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        frames = os.path.join(session, "frames")
        os.makedirs(frames, exist_ok=True)
        return session, frames

    def _record_camera_geometry(self, frame):
        """Capture the source geometry once, so the saved 64x64 arrays can be
        interpreted later. The square centre crop keeps a different share of the
        horizontal view depending on aspect ratio (0.5625 at 16:9, 0.75 at 4:3),
        which shifts effective FOV even when the lens has not changed.
        """
        height, width = frame.shape[:2]
        side = min(height, width)
        self._camera_geometry = {
            "source_resolution": [int(width), int(height)],
            "source_aspect": round(width / height, 4),
            "crop_box_xywh": [(width - side) // 2, (height - side) // 2, side, side],
            "crop_fraction_of_width": round(side / width, 4),
            "crop_fraction_of_height": round(side / height, 4),
            "resize_to": [64, 64],
            "interpolation": "INTER_AREA",
            "rotate_180": self.rotate_180,
        }
        self.get_logger().warn(
            f"camera geometry: {width}x{height}, centre crop keeps "
            f"{self._camera_geometry['crop_fraction_of_width'] * 100:.1f}% of horizontal FOV "
            "-- record this before comparing against sim frames"
        )
        self._write_session_meta()

    def _write_session_meta(self):
        meta = {
            "session_dir": self.session_dir,
            "started_at": self._started_at,
            "route": self.route_name,
            "route_one_hot": ROUTE_ONE_HOT[self.route_name],
            "robot_id": self.robot_id,
            "image_topic": f"/robomaster_{self.robot_id}/camera_0/image_raw",
            "cmd_topic": f"/robomaster_{self.robot_id}/cmd_vel",
            "model_sha256": getattr(self, "model_sha256", None),
            "control": {
                "rate_hz": self.rate_hz,
                "frame_timeout_s": self.frame_timeout_s,
                "scale_linear": self.scale_linear,
                "scale_angular": self.scale_angular,
                "cap_linear": self.cap_linear,
                "cap_angular": self.cap_angular,
                "use_header_stamp": self.use_header_stamp,
            },
            "camera_geometry": getattr(self, "_camera_geometry", None),
            "sim_camera": self.manifest.get("sim_camera"),
            "steps_logged": self._step,
            "status_counts": dict(self._status_counts),
            "ended_at": self._ended_at,
        }
        try:
            with open(os.path.join(self.session_dir, "session_meta.json"), "w") as handle:
                json.dump(meta, handle, indent=2)
        except Exception as exc:
            self.get_logger().warn(f"could not write session_meta.json: {exc}")

    def _hand_logs_to_host_user(self):
        """The container runs as root over a bind mount, so hand the session back
        to whoever owns the log root; otherwise copying it off the host needs sudo.
        """
        try:
            target = os.stat(os.path.dirname(self.session_dir.rstrip("/")))
            if os.geteuid() != 0 or target.st_uid == 0:
                return
            for root, dirs, files in os.walk(self.session_dir):
                for name in dirs + files:
                    os.chown(os.path.join(root, name), target.st_uid, target.st_gid)
            os.chown(self.session_dir, target.st_uid, target.st_gid)
        except Exception as exc:
            self.get_logger().warn(f"could not hand logs to host user: {exc}")

    # ---------- callbacks ----------

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return
        with self._lock:
            self._latest_frame = frame
            self._latest_rx_time = time.monotonic()
            self._latest_header_stamp = msg.header.stamp
        if self._camera_geometry is None:
            self._record_camera_geometry(frame)

    def _arm_cb(self, request, response):
        self.armed = bool(request.data)
        if not self.armed:
            self._publish_zero()
        state = "ARMED" if self.armed else "DISARMED"
        self.get_logger().warn(f"{state} by service call")
        response.success = True
        response.message = state
        return response

    # ---------- control loop ----------

    def _orient(self, frame_rgb):
        if self.rotate_180:
            return cv2.rotate(frame_rgb, cv2.ROTATE_180)
        return frame_rgb

    def _preprocess(self, frame_rgb):
        height, width = frame_rgb.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        square = frame_rgb[top:top + side, left:left + side]
        small = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
        normalised = small.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).contiguous()
        return normalised, tensor

    def _control_step(self):
        if self._shutting_down:
            return
        try:
            with self._lock:
                # cv_bridge allocates a fresh array per message, so holding the
                # reference is safe and avoids copying 6 MB every step.
                frame = self._latest_frame
                rx_time = self._latest_rx_time
                header_stamp = self._latest_header_stamp

            if frame is None:
                self._publish_zero()
                self.get_logger().warn("no camera frame received yet; commanding zero", throttle_duration_sec=2.0)
                self._log_row(float("nan"), float("nan"), None, None, None, "no_frame")
                return

            frame_age = time.monotonic() - rx_time
            header_age = float("nan")
            if header_stamp is not None:
                header_age = self.get_clock().now().nanoseconds / 1e9 - (
                    header_stamp.sec + header_stamp.nanosec / 1e9
                )
            watchdog_age = header_age if self.use_header_stamp else frame_age

            frame = self._orient(frame)
            normalised, tensor = self._preprocess(frame)
            with torch.no_grad():
                action = self.model(tensor, self.route_tensor)
            raw = [float(v) for v in action.squeeze(0)]

            scaled = (
                raw[0] * self.scale_linear,
                raw[1] * self.scale_linear,
                raw[2] * self.scale_angular,
            )
            clamped = (
                self._clamp(scaled[0], self.cap_linear),
                self._clamp(scaled[1], self.cap_linear),
                self._clamp(scaled[2], self.cap_angular),
            )

            stale = watchdog_age > self.frame_timeout_s
            if stale:
                self.get_logger().warn(
                    f"stale frame ({watchdog_age:.3f}s > {self.frame_timeout_s}s); commanding zero",
                    throttle_duration_sec=1.0,
                )

            if self.armed and not stale:
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.angular.z = clamped
                self.cmd_pub.publish(twist)
                status = "armed"
            else:
                self._publish_zero()
                status = "stale" if stale else "disarmed"

            self._save_frames(normalised, frame)
            self._log_row(frame_age, header_age, raw, scaled, clamped, status)
            self._step += 1

        except Exception as exc:
            self._publish_zero()
            self.get_logger().error(f"control step failed, commanding zero: {exc}")

    @staticmethod
    def _clamp(value, limit):
        return max(-limit, min(limit, value))

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def _save_frames(self, normalised, full_frame_rgb):
        jobs = [(os.path.join(self.frames_dir, f"obs_{self._step:06d}.npy"), normalised, "npy")]
        now = time.monotonic()
        if now - self._last_jpeg_time >= self.jpeg_period_s:
            jobs.append((os.path.join(self.frames_dir, f"full_{self._step:06d}.jpg"), full_frame_rgb, "jpg"))
            self._last_jpeg_time = now
        for job in jobs:
            try:
                self._write_queue.put_nowait(job)
            except queue.Full:
                self.get_logger().warn("frame writer backlogged; dropped a frame", throttle_duration_sec=5.0)

    def _writer_loop(self):
        while True:
            job = self._write_queue.get()
            if job is None:
                break
            path, array, kind = job
            try:
                if kind == "npy":
                    np.save(path, array)
                else:
                    cv2.imwrite(path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            except Exception as exc:  # keep the control loop alive regardless
                self.get_logger().error(f"failed writing {path}: {exc}")
            finally:
                self._write_queue.task_done()

    def _log_row(self, frame_age, header_age, raw, scaled, clamped, status):
        blanks = ["", "", ""]
        self._status_counts[status] += 1
        self._csv.writerow(
            [f"{time.time():.6f}", self._step, f"{frame_age:.4f}", f"{header_age:.4f}"]
            + ([f"{v:.6f}" for v in raw] if raw else blanks)
            + ([f"{v:.6f}" for v in scaled] if scaled else blanks)
            + ([f"{v:.6f}" for v in clamped] if clamped else blanks)
            + [int(self.armed), self.route_name, status]
        )
        self._csv_file.flush()

    # ---------- teardown ----------

    def safe_stop(self):
        """Command zero repeatedly; safe to call more than once."""
        self._shutting_down = True
        self.armed = False
        for _ in range(ZERO_PUBLISH_REPEATS):
            try:
                self._publish_zero()
            except Exception:
                break
            time.sleep(0.02)
        try:
            self._write_queue.put_nowait(None)
            self._writer.join(timeout=5.0)
        except Exception:
            pass
        try:
            self._csv_file.close()
        except Exception:
            pass
        self._ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_session_meta()
        self._hand_logs_to_host_user()


def main():
    rclpy.init()
    node = None
    try:
        node = PolicyRunner()
    except SystemExit as exc:
        print(f"{exc}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    def on_signal(_signum, _frame):
        node.get_logger().warn("signal received; zeroing command and shutting down")
        node.safe_stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f"unhandled exception: {exc}")
    finally:
        node.safe_stop()
        node.get_logger().info(f"logs written to {node.session_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
