"""Record a research-grade video of the composed race-v8 system.

Runs the deployed architecture — scout camera -> frozen JEPA encoder -> z_t on
the wire -> race-v8 route head -> 1-bit command -> frozen stage-1.5 executor —
headless, and renders each control step to a 1920x1080 frame:

  left   : overhead cinematic camera over the arena (live ray-traced render)
  right  : the science panel — scout view (the message source), navigator view
           (the policy input), the head's decode with confidence and verdict,
           live telemetry, and a trajectory mini-map with the hazard slab.

The slab side ALTERNATES deterministically across episodes, so the video's
argument is visual: the hazard flips, the decoded command flips, the route
flips — perception on one robot steering another through a learned 1-bit
channel. Frames stream straight into ffmpeg (imageio), no intermediate PNGs.

    source setup/env.sh
    CUDA_VISIBLE_DEVICES=1 python spike/record_video.py \
        --out plots/v8_live_demo.mp4 --episodes 4

A probe still of frame 10 is saved next to the mp4 for a quick layout check.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="plots/v8_live_demo.mp4")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=0,
                    help="hard frame cap (0 = run all episodes to completion)")
parser.add_argument("--fps", type=int, default=20,
                    help="playback fps; control runs at 10 Hz, so 20 = 2x real time")
parser.add_argument("--executor", type=str, default="runs/route_obey_v6/cont.pt")
parser.add_argument("--head", type=str, default="runs/race_v8/z_t_s3.pt")
parser.add_argument("--condition", type=str, default="z_t",
                    choices=["none", "z_t", "oracle"])
parser.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_pixels.pt")
parser.add_argument("--comm_radius", type=float, default=12.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import traceback


def _die_loudly(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _die_loudly

import imageio.v3 as iio
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM, N_ACTIONS  # noqa: E402
from chokepoint.env import ChokepointEnv, ChokepointEnvCfg  # noqa: E402
from chokepoint.geometry import chokepoint_grid  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.message_bus import LatentBroadcast, OracleBroadcast  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402
from chokepoint.route_head import RouteHead  # noqa: E402

LEARNER, BEACON = "navigator", "scout"
ROUTE_DIM = 2
WIRE = LATENT_DIM + 2
DECIDE_STEP = 2

# ---- frame layout ------------------------------------------------------------
W, H = 1920, 1080
CAM = 1080                      # overhead render is square, fills the left
PANEL_X = CAM + 40              # sidebar origin
PANEL_W = W - CAM - 80
BG = (17, 19, 24)
FG = (235, 235, 235)
DIM = (150, 155, 165)
RED = (196, 78, 82)             # thesis red (matches the figures)
BLUE = (76, 114, 176)
GREEN = (85, 168, 104)
GOLD = (221, 170, 51)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}", size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_H1, F_BODY, F_SMALL = font(30, True), font(26, True), font(22), font(17)


class MiniMap:
    """Static wall raster + per-frame slab/goal/trail overlay."""

    def __init__(self, grid: "np.ndarray", cell: float, size_px: int = 340):
        self.n = grid.shape[0]
        self.extent = self.n * cell
        self.px = size_px
        base = Image.new("RGB", (size_px, size_px), (26, 29, 36))
        d = ImageDraw.Draw(base)
        s = size_px / self.n
        for r in range(self.n):
            for c in range(self.n):
                if grid[r, c]:
                    # grid_to_world: x = (c - n/2 + .5)*cell, y = (n/2 - r - .5)*cell
                    x0 = c * s
                    y0 = r * s
                    d.rectangle([x0, y0, x0 + s, y0 + s], fill=(110, 116, 132))
        self.base = base

    def world_to_px(self, x: float, y: float):
        u = (x + self.extent / 2) / self.extent * self.px
        v = (self.extent / 2 - y) / self.extent * self.px
        return u, v

    def render(self, slab_aabb, goal_xy, trail, robot_xy, scout_xy):
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        x0, x1, y0, y1 = slab_aabb
        u0, v0 = self.world_to_px(x0, y1)
        u1, v1 = self.world_to_px(x1, y0)
        d.rectangle([u0, v0, u1, v1], fill=RED)
        if len(trail) > 1:
            pts = [self.world_to_px(x, y) for x, y in trail]
            d.line(pts, fill=(90, 200, 250), width=3)
        gu, gv = self.world_to_px(*goal_xy)
        d.ellipse([gu - 7, gv - 7, gu + 7, gv + 7], outline=GOLD, width=3)
        su, sv = self.world_to_px(*scout_xy)
        d.ellipse([su - 5, sv - 5, su + 5, sv + 5], fill=(200, 200, 90))
        ru, rv = self.world_to_px(*robot_xy)
        d.ellipse([ru - 6, rv - 6, ru + 6, rv + 6], fill=(90, 200, 250),
                  outline=(255, 255, 255), width=2)
        return img


def upscale(rgb01: np.ndarray, size: int) -> Image.Image:
    """64x64 float [0,1] camera tensor -> crisp nearest-neighbour thumbnail."""
    img = Image.fromarray((rgb01 * 255).astype(np.uint8))
    return img.resize((size, size), Image.NEAREST)


def main():
    cfg = ChokepointEnvCfg()
    cfg.scene.num_envs = 1
    cfg.success_agents = [LEARNER]
    cfg.force_slab_top = True  # episode 1; alternated below

    # Cinematic overhead camera, top-down: quaternion (0,1,0,0) maps camera
    # +X->world +X (image right = east) and +Y->world -Y (image up = north),
    # matching the mini-map orientation. FOV(f=12, aperture 20.955) ~ 82 deg,
    # so height ~ extent/1.7 frames the whole arena.
    grid = chokepoint_grid(cfg.map_seed)
    extent = grid.shape[0] * cfg.cell
    cfg.scene.cam_overhead = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, extent / 2.2 + 0.5), rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
        update_period=0,
        height=CAM,
        width=CAM,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 40.0)),
    )

    env = ChokepointEnv(cfg)
    device = env.device
    obs, _ = env.reset()

    ckpt = torch.load(args.jepa_ckpt, map_location=device)
    encoder = PixelEncoder(ckpt["config"]["latent_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    executor = AttentionReceiver(
        encoder, broadcast_dim=0, latent_dim=LATENT_DIM, route_dim=ROUTE_DIM
    ).to(device)
    executor.load_state_dict(torch.load(args.executor, map_location=device)["policy"])
    executor.eval()

    head = RouteHead(WIRE).to(device)
    head.load_state_dict(torch.load(args.head, map_location=device)["head"])
    head.eval()

    if args.condition == "oracle":
        bus = OracleBroadcast(comm_radius=args.comm_radius,
                              broadcast_dim=LATENT_DIM, anchored=True)
    elif args.condition == "z_t":
        bus = LatentBroadcast(encoder, comm_radius=args.comm_radius,
                              broadcast_dim=LATENT_DIM, anchored=True)
    else:
        bus = None

    def msg_vec():
        if bus is None:
            return torch.zeros(1, WIRE, device=device)
        messages, mask = bus.deliver(env)[LEARNER]
        return torch.nan_to_num(messages[:, 0, :] * mask[:, 0:1].float())

    empty_msg = torch.zeros(1, 0, 1, device=device)
    empty_mask = torch.zeros(1, 0, device=device)
    zero_scout = torch.zeros(1, N_ACTIONS, device=device)
    goal = env._goal_pos[LEARNER][0].tolist()
    scout_xy = env._local_pos(BEACON)[0].tolist()
    minimap = MiniMap(grid, cfg.cell)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    # settle the renderer (denoiser/exposure) before the first recorded frame
    for _ in range(5):
        obs, _, _, _, _ = env.step({LEARNER: zero_scout, BEACON: zero_scout})

    episode, step, frames = 1, 0, 0
    route_top, conf, committed, touched = None, None, None, False
    route = torch.zeros(1, ROUTE_DIM, device=device)
    trail: list = []
    outcome_text, outcome_color, outcome_hold = "", FG, 0

    def slab_side():
        return "TOP" if env._slab_top[0] else "BOTTOM"

    print(f"[rec] recording {args.episodes} episodes -> {out}")
    while episode <= args.episodes:
        # alternate the NEXT episode's slab so the decision flips on camera
        env.cfg.force_slab_top = episode % 2 == 0

        if step == DECIDE_STEP:
            with torch.no_grad():
                logits, _ = head(msg_vec())
                p = torch.softmax(logits, dim=-1)[0]
            route_top = bool(p[0] > p[1])
            conf = float(p.max())
            route = torch.zeros(1, ROUTE_DIM, device=device)
            route[0, 0 if route_top else 1] = 1.0
            print(f"[ep {episode}] slab {slab_side()} -> head commands "
                  f"{'TOP' if route_top else 'BOTTOM'} (p={conf:.2f})")

        rgb = obs[LEARNER].permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            action = executor.actor(executor.features(rgb, empty_msg, empty_mask, route))
        nav_view = obs[LEARNER][0].cpu().numpy()
        scout_view = obs[BEACON][0].cpu().numpy()
        obs, _, term, tout, _ = env.step(
            {LEARNER: action.clamp(-1, 1), BEACON: zero_scout}
        )
        step += 1

        pos = env._local_pos(LEARNER)[0]
        x, y = float(pos[0]), float(pos[1])
        trail.append((x, y))
        in_haz = bool(env._in_hazard(LEARNER)[0])
        touched |= in_haz
        if committed is None and x > -3.0 and abs(y) > 0.5:
            committed = "top" if y > 0 else "bottom"

        # ---- compose the frame ----------------------------------------------
        top_cam = env.scene["cam_overhead"].data.output["rgb"][0].cpu().numpy()
        frame = Image.new("RGB", (W, H), BG)
        frame.paste(Image.fromarray(top_cam.astype(np.uint8)), (0, 0))
        d = ImageDraw.Draw(frame)

        yy = 28
        d.text((PANEL_X, yy), "Latent Telepathy — Tier 2", font=F_TITLE, fill=FG)
        yy += 42
        d.text((PANEL_X, yy),
               "scout z_t  →  route head  →  frozen executor  (Isaac Sim)",
               font=F_SMALL, fill=DIM)
        yy += 40
        d.text((PANEL_X, yy), f"episode {episode}   step {step:3d}   "
               f"(2× real time)", font=F_BODY, fill=DIM)
        yy += 44

        side = slab_side() if step > 0 else "?"
        d.text((PANEL_X, yy), "hazard slab:", font=F_BODY, fill=FG)
        d.text((PANEL_X + 170, yy), f"{side} corridor", font=F_H1, fill=RED)
        yy += 42
        if route_top is not None:
            cmd = "TOP" if route_top else "BOTTOM"
            ok = route_top != bool(env._slab_top[0])
            d.text((PANEL_X, yy), "head decodes:", font=F_BODY, fill=FG)
            d.text((PANEL_X + 170, yy), f"{cmd}  (p={conf:.2f})", font=F_H1,
                   fill=GREEN if ok else RED)
            yy += 36
            d.text((PANEL_X + 170, yy),
                   "= safe corridor" if ok else "= SLABBED corridor",
                   font=F_SMALL, fill=GREEN if ok else RED)
        else:
            d.text((PANEL_X, yy), "head decodes:  …", font=F_BODY, fill=DIM)
            yy += 36
        yy += 40

        thumb = 230
        frame.paste(upscale(scout_view, thumb), (PANEL_X, yy))
        frame.paste(upscale(nav_view, thumb), (PANEL_X + thumb + 30, yy))
        d.text((PANEL_X, yy + thumb + 8), "scout view (message source)",
               font=F_SMALL, fill=DIM)
        d.text((PANEL_X + thumb + 30, yy + thumb + 8), "navigator view (policy input)",
               font=F_SMALL, fill=DIM)
        yy += thumb + 44

        aabb = torch.as_tensor(
            env._aabb_top if env._slab_top[0] else env._aabb_bot
        ).reshape(-1)[:4].tolist()
        mm = minimap.render(aabb, goal[:2], trail, (x, y), scout_xy)
        frame.paste(mm, (PANEL_X, yy))
        d.text((PANEL_X + minimap.px + 20, yy), "trajectory", font=F_SMALL, fill=DIM)
        st_y = yy + 30
        status = [
            (f"pos ({x:+.1f}, {y:+.1f})", DIM),
            (f"goal {float(torch.norm(pos[:2] - torch.tensor(goal[:2], device=device))):.1f} m", DIM),
            (f"corridor: {committed or '—'}", FG),
            ("IN HAZARD" if in_haz else "clear", RED if in_haz else GREEN),
        ]
        for txt, col in status:
            d.text((PANEL_X + minimap.px + 20, st_y), txt, font=F_SMALL, fill=col)
            st_y += 26

        if outcome_hold > 0:
            d.text((PANEL_X, H - 70), outcome_text, font=F_H1, fill=outcome_color)
            outcome_hold -= 1

        arr = np.asarray(frame)
        writer.append_data(arr)
        frames += 1
        if frames == 10:
            iio.imwrite(str(out.with_suffix(".probe.png")), arr)
        if frames % 100 == 0:
            print(f"[rec] {frames} frames…")
        if args.max_steps and frames >= args.max_steps:
            break

        if bool((term[LEARNER] | tout[LEARNER])[0]):
            ok = bool(term[LEARNER][0])
            outcome_text = (f"ep {episode}: REACHED GOAL — "
                            f"{'clean' if not touched else 'crossed slab'}"
                            if ok else f"ep {episode}: timed out")
            outcome_color = GREEN if ok and not touched else RED
            outcome_hold = args.fps  # show the banner for ~1 s
            print(f"[ep {episode}] {outcome_text} ({step} steps)")
            episode += 1
            step, committed, touched = 0, None, False
            route_top, conf = None, None
            route = torch.zeros(1, ROUTE_DIM, device=device)
            trail = []

    writer.close()
    print(f"[rec] wrote {frames} frames -> {out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
