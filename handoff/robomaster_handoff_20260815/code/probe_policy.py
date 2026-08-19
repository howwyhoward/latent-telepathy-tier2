#!/usr/bin/env python3
"""Run the exported policy offline over the captured real frames and write a CSV.

This exists because the bench sessions in deploy/logs were driven by fake_camera.py,
which publishes a uniform grey field, so they say nothing about behaviour on real
imagery. The pad captures are the only real frames we have, and feeding them through
the same TorchScript artifact the robot loads is the cheapest honest answer to "what
does the policy do when it sees the lab?".

Inputs are the *_64.npy arrays written by fov_check.py, which are already in the
manifest's contract: (64,64,3) float32 RGB in [0,1], centre-cropped and INTER_AREA
resized, with the inverted camera mount corrected.

Usage: python3 ~/deploy/probe_policy.py [--out policy_probe.csv]
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SATURATED = 0.99

REFERENCES = [
    ("ref_grey_0.5", np.full((64, 64, 3), 0.5, np.float32)),
    ("ref_grey_128", np.full((64, 64, 3), 128.0 / 255.0, np.float32)),
    ("ref_black", np.zeros((64, 64, 3), np.float32)),
    ("ref_white", np.ones((64, 64, 3), np.float32)),
    ("ref_noise", np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)),
]


def load_captures(directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*_64.npy"))):
        tag = os.path.basename(path).split("_640x360")[0]
        out.append((tag, np.load(path)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(HERE, "policy_deploy.pt"))
    parser.add_argument("--captures", default=os.path.join(HERE, "fov_check"))
    parser.add_argument("--out", default=os.path.join(HERE, "policy_probe.csv"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = torch.jit.load(args.model, map_location="cpu").eval()
    routes = {"top": torch.tensor([[1.0, 0.0]]), "bottom": torch.tensor([[0.0, 1.0]])}

    rows = []
    for tag, frame in REFERENCES + load_captures(args.captures):
        chw = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))[None].float()
        row = {"input": tag, "is_reference": tag.startswith("ref_"),
               "frame_mean": round(float(frame.mean()), 4),
               "frame_std": round(float(frame.std()), 4)}
        for name, route in routes.items():
            action = model(chw, route)[0].numpy()
            for i, value in enumerate(action):
                row[f"{name}_a{i}"] = round(float(value), 4)
            row[f"{name}_saturated"] = int(np.sum(np.abs(action) >= SATURATED))
        rows.append(row)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    real = [r for r in rows if not r["is_reference"]]
    channels = 3 * 2 * len(real)
    saturated = sum(r["top_saturated"] + r["bottom_saturated"] for r in real)
    print(f"wrote {args.out}  ({len(rows)} inputs, {len(real)} real frames)")
    print(f"saturated output channels on real frames: {saturated}/{channels} "
          f"({100.0 * saturated / channels:.0f}%)")
    for name in ("top", "bottom"):
        for i in range(3):
            values = [r[f"{name}_a{i}"] for r in real]
            print(f"  real frames, route {name:6s} a{i}: min {min(values):+.4f}  max {max(values):+.4f}")


if __name__ == "__main__":
    main()
