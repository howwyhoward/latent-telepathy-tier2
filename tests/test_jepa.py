"""Unit tests for the pixel JEPA port (no Isaac required)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM  # noqa: E402
from chokepoint.geometry import (  # noqa: E402
    generate_chokepoint_map,
    obstacle_rects,
    sample_free_positions,
)
from chokepoint.jepa import (  # noqa: E402
    JEPA,
    SEG_CLASSES,
    SEG_RES,
    SegDecoder,
    jepa_loss,
    make_class_weights,
    reconstruction_loss,
)

B = 6


def batch():
    rgb_t = torch.rand(B, 3, 64, 64)
    a = torch.rand(B, 3) * 2 - 1
    rgb_n = torch.rand(B, 3, 64, 64)
    return rgb_t, a, rgb_n


def test_shapes():
    m = JEPA()
    z_pred, z_target, z_t = m(*batch())
    assert z_pred.shape == z_target.shape == z_t.shape == (B, LATENT_DIM)
    d = SegDecoder()
    logits = d(z_t)
    assert logits.shape == (B, len(SEG_CLASSES), SEG_RES, SEG_RES)


def test_target_gets_no_grad():
    m = JEPA()
    z_pred, z_target, z_t = m(*batch())
    (jepa_loss(z_pred, z_target) + z_t.sum()).backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert any(p.grad is not None for p in m.encoder.parameters())
    assert any(p.grad is not None for p in m.predictor.parameters())


def test_ema_moves_target_toward_online():
    m = JEPA()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.target_encoder.parameters()]
    m.update_target(tau=0.9)
    for b, tp, op in zip(before, m.target_encoder.parameters(), m.encoder.parameters()):
        assert torch.allclose(tp, 0.9 * b + 0.1 * op, atol=1e-5)


def test_jepa_loss_bounds():
    # normalized MSE = 2 - 2cos in [0, 4]; identical inputs -> 0
    z = torch.randn(B, LATENT_DIM)
    assert jepa_loss(z, z).item() < 1e-6
    val = jepa_loss(torch.randn(B, LATENT_DIM), torch.randn(B, LATENT_DIM)).item()
    assert 0.0 <= val <= 4.0


def test_class_weights_upweight_rare_classes():
    w = make_class_weights(10.0)
    assert w[SEG_CLASSES.index("hazard")] == 10.0
    assert w[SEG_CLASSES.index("goal")] == 10.0
    assert w[SEG_CLASSES.index("agent")] == 10.0
    assert w[SEG_CLASSES.index("background")] == 1.0
    assert w[SEG_CLASSES.index("wall")] == 1.0


def test_weighted_recon_prefers_getting_hazard_right():
    w = make_class_weights(10.0)
    target = torch.full((B, SEG_RES, SEG_RES), SEG_CLASSES.index("hazard"))
    good = torch.zeros(B, len(SEG_CLASSES), SEG_RES, SEG_RES)
    good[:, SEG_CLASSES.index("hazard")] = 5.0
    bad = torch.zeros_like(good)
    bad[:, SEG_CLASSES.index("background")] = 5.0
    assert reconstruction_loss(good, target, w) < reconstruction_loss(bad, target, w)


def test_spawn_sampler_avoids_obstacles():
    grid = generate_chokepoint_map(np.random.default_rng(2)).grid
    cell = 0.5
    pts = sample_free_positions(grid, cell, 500, np.random.default_rng(0), margin=0.22)
    assert pts.shape == (500, 2)
    for x0, x1, y0, y1 in obstacle_rects(grid, cell):
        inside = (
            (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
            & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        )
        assert not inside.any()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
