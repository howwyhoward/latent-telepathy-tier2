# Bug: `camera_source` timer period truncates to zero (busy-polls, caps frame rate)

**Component:** `robomaster/cam_driver/camera/src/camera_source.cpp:85`
**Severity:** High — pegs a CPU core on the Jetson and makes `framerate` a no-op
**Found:** 2026-08-12, RoboMaster S1 + Jetson Orin, ROS 2 Humble, `cam_driver_dnv:latest`

## Symptom

`camera_source` publishes `image_raw` at ~2.8 Hz at 1920x1080 instead of the
requested 15 Hz, and the process consumes ~100% of one core while doing it. The
`framerate` parameter has no observable effect at any value.

## Cause

```cpp
timer_frames_ = rclcpp::create_timer(
                    this,
                    get_clock(),
                    rclcpp::Duration(0, (int) (1.f / video_options.frameRate) * 1e9),
                    std::bind(&CameraSource::timer_frames_callback, this)
                );
```

The cast binds tighter than the multiply, so the division is truncated to an
integer *before* being scaled to nanoseconds:

| `frameRate` | `1.f / frameRate` | `(int)(...)` | `* 1e9` | timer period |
|---|---|---|---|---|
| 15.0 | 0.0667 | 0 | 0 | 0 ns |
| 30.0 | 0.0333 | 0 | 0 | 0 ns |
| 1.0 | 1.0 | 1 | 1e9 | 1 s (accidentally correct) |

For any `frameRate > 1.0` the period is zero, so the timer refires immediately
on every executor pass. The callback blocks in `stream_->Capture(&nextFrame, 1000)`,
so the observed publish rate is not a paced rate at all — it is the raw
capture/convert/publish throughput of the node, with the executor spinning flat
out in between. That explains both the CPU burn and why the measured rate scales
with per-frame cost (resolution) rather than with the `framerate` parameter.

## Fix

Scale to nanoseconds first, then convert:

```diff
--- a/camera/src/camera_source.cpp
+++ b/camera/src/camera_source.cpp
@@
     timer_frames_ = rclcpp::create_timer(
                         this,
                         get_clock(),
-                        rclcpp::Duration(0, (int) (1.f / video_options.frameRate) * 1e9),
+                        rclcpp::Duration(std::chrono::nanoseconds(
+                            static_cast<int64_t>(1e9 / video_options.frameRate))),
                         std::bind(&CameraSource::timer_frames_callback, this)
                     );
```

The `std::chrono::nanoseconds` constructor is preferred over
`rclcpp::Duration(0, nsec)` because the two-argument form takes a `uint32_t`
nanosecond field, which silently wraps for periods of 1 s or longer (i.e. for
`frameRate < 1.0`). Requires `#include <chrono>`.

Guarding the parameter is also worth doing, since `frameRate <= 0` currently
yields a division by zero:

```cpp
if (video_options.frameRate <= 0.0f) {
    RCLCPP_WARN(get_logger(), "framerate %.1f invalid; defaulting to 30", video_options.frameRate);
    video_options.frameRate = 30.0f;
}
```

## Verification

After rebuilding, `ros2 topic hz /robomaster_1/camera_0/image_raw` should track
the `framerate` parameter, and `top` should show `camera_source` well below one
full core at 15 Hz.

## Note on the workaround used in the meantime

Because the callback blocks on capture, throughput is bounded by per-frame cost.
Reducing the requested resolution therefore raises the achievable rate without
touching the code — this is the workaround in use for the current plumbing test,
not a fix.

## Second issue: `image_raw` is rotated 180 degrees, with no way to correct it

The camera module is mounted inverted, so `image_raw` is published upside down —
floor at the top of the frame. Verified by capturing a still and rotating it: the
rotated copy shows the chair, pedestal, light switch and robots in their true
positions. `camera_proc` does not correct it either, so `image_proc` is inverted
too.

`camera_source` exposes no way to fix this. `jetson-utils` already supports it via
`videoOptions::flipMethod`, which the node never sets or declares, so it stays at
its default:

```cpp
declare_parameter("width", 1920);
declare_parameter("height", 1080);
declare_parameter("framerate", 30.0);
declare_parameter("loop", 0);
// no flip parameter
```

Suggested fix — declare it and map the string onto the existing enum, which costs
nothing because `nvvidconv flip-method` is already in the pipeline:

```cpp
declare_parameter("flip_method", "rotate-180");
std::string flip_str;
get_parameter("flip_method", flip_str);
video_options.flipMethod = videoOptions::FlipMethodFromStr(flip_str.c_str());
```

`rotate-180` is the correct default for this chassis. Any downstream consumer is
currently obliged to rotate in software, which every consumer has to rediscover
independently — for a learned policy trained on upright images, silently inverted
input is a correctness bug, not a cosmetic one.

## Third issue: `camera_stream_0.service` is broken by design and hides it

The unit is `enabled`, so it starts on every boot, and its command is:

```
/usr/bin/docker run --rm --runtime nvidia --net=host --ipc=host \
  -v /tmp/argus_socket:/tmp/argus_socket --hostname robomaster-1 \
  cam_driver_dnv:latest /bin/bash -c '. install/setup.bash && \
  while true; do ros2 launch src/camera/launch/camera.launch.py; done'
```

Two problems compound:

**It cannot work on this image.** The `docker run` has no
`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0`, so the NVIDIA
GStreamer plugins fail to load ("cannot allocate memory in static TLS block") and
`camera_source` exits with "failed to open video source". Its journal shows this
happening continuously.

**The `while true` loop hides the failure.** Because the loop restarts the launch
file forever, the unit stays `active (running)` and never reports a failure, so
nothing surfaces. Observed for 48 minutes straight with no camera publishing at
any point. Meanwhile each iteration takes another swing at the CSI sensor, so it
also kills any manually launched `camera_source`: ours ran fine at 640x360, then
exited 255 with no error once the loop's attempts contended for Argus.

The end state is confusing to diagnose, because `camera_proc` survives inside the
same container and *subscribes* to `image_raw`. So `ros2 topic list` shows
`image_raw`, and only `ros2 topic info -v` reveals `Publisher count: 0`,
`Subscription count: 1`.

Suggested fix: add `-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0`
to the unit, drop the `while true` in favour of `Restart=on-failure` so failures
are visible in `systemctl status`, and set the resolution parameters there rather
than relying on the launch file's 1080p default.

**Trap for anyone working around this:** `systemctl stop camera_stream_0` does not
stop the container. The unit's main process is the `docker run` *client*; killing
it leaves the container running under the daemon. It must be removed explicitly:

```bash
docker ps --filter ancestor=cam_driver_dnv:latest -q | xargs -r docker rm -f
```

## Fourth issue: system clock

The Jetson's clock is roughly 10 months behind real time and moves while running —
`docker ps -a` reported one container as "Created 5 minutes ago" and
"Exited 48 minutes ago" simultaneously. Message header stamps are therefore
unusable for freshness checks by any consumer. Needs chrony or NTP.

## Unrelated issue found in the same session

`camera_proc` crashes at startup when `camera/cfg/front_camera.yaml` carries five
`plumb_bob` distortion coefficients, because `cv::fisheye::estimateNewCameraMatrixForUndistortRectify`
asserts on exactly four:

```
Assertion failed) D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || CV_64F))
```

Either the shipped calibration file or the undistortion model is wrong — the node
runs fisheye math against a pinhole calibration. Worked around by bind-mounting a
4-coefficient YAML over the shipped one.
