"""Shared frame labels + loader for the 15 Aug handoff captures."""

from pathlib import Path

import numpy as np
import torch

HANDOFF = Path(__file__).resolve().parent / "robomaster_handoff_20260815"

# label 1 = red pad (hazard analog) visible, 0 = not; None = health check only
FRAMES = {
    "ctrl_a": 0, "ctrl_b": 0,
    "red_100": 1, "red_150": 1, "red_200": 1, "red_280": 1,
    "red_l_280": 1, "red_r_280": 1, "rg_split_280": 1,
    "green_200": 0, "green_280": 0,
    "calib_tape_150": None, "calib_tape_280": None,
}


def load_real_frames():
    names, imgs, labels = [], [], []
    for stem, lab in FRAMES.items():
        matches = sorted(HANDOFF.glob(f"pad_captures/{stem}_*_64.npy"))
        assert len(matches) == 1, f"{stem}: {matches}"
        a = np.load(matches[0])
        assert a.shape == (64, 64, 3) and a.dtype == np.float32
        names.append(stem)
        imgs.append(torch.from_numpy(a).permute(2, 0, 1))
        labels.append(lab)
    return names, torch.stack(imgs), labels
