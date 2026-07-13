"""Unit tests for the ported AttentionReceiver — no Isaac required.

Re-establishes the Tier 1 guarantees: frozen encoder, permutation-invariant
masked pooling, exact-zero contribution from masked slots, and the zero-init
value path (every condition starts as the `none` policy).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chokepoint.receiver import AttentionReceiver

LATENT = 64
B, N, D_MSG = 6, 3, 66  # 3 neighbor slots, anchored 64-D channel


class TinyEncoder(nn.Module):
    """Stand-in for the frozen JEPA encoder: (B, 3, H, W) -> (B, LATENT)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 5, stride=4)
        self.head = nn.LazyLinear(LATENT)

    def forward(self, x):
        h = torch.relu(self.conv(x)).flatten(1)
        return self.head(h)


def make_receiver(broadcast_dim=D_MSG, **kw):
    enc = TinyEncoder()
    enc(torch.zeros(1, 3, 64, 64))  # materialize lazy layer before freezing
    return AttentionReceiver(enc, broadcast_dim=broadcast_dim, latent_dim=LATENT, **kw)


def rand_inputs(broadcast_dim=D_MSG):
    rgb = torch.rand(B, 3, 64, 64)
    msgs = torch.randn(B, N, broadcast_dim) * 10  # latent-scale magnitudes
    mask = torch.ones(B, N, dtype=torch.int8)
    return rgb, msgs, mask


def test_encoder_is_frozen():
    r = make_receiver()
    assert all(not p.requires_grad for p in r.encoder.parameters())
    rgb, msgs, mask = rand_inputs()
    logits, value = r(rgb, msgs, mask)
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is None for p in r.encoder.parameters())


def test_zero_init_value_path_starts_as_none_policy():
    r = make_receiver()
    rgb, msgs, mask = rand_inputs()
    logits_with, _ = r(rgb, msgs, mask)
    logits_without, _ = r(rgb, torch.zeros_like(msgs), torch.zeros_like(mask))
    assert torch.allclose(logits_with, logits_without, atol=1e-6)


def test_masked_slots_contribute_exactly_zero():
    r = make_receiver()
    # break the zero-init so messages CAN matter, then check masking
    nn.init.normal_(r.v.weight)
    rgb, msgs, mask = rand_inputs()
    mask[:, 1] = 0
    ref = r.pool_neighbors(r.ego_proj(r.encode_ego(rgb)), msgs, mask)
    msgs2 = msgs.clone()
    msgs2[:, 1] = torch.randn_like(msgs2[:, 1]) * 100  # garbage in masked slot
    out = r.pool_neighbors(r.ego_proj(r.encode_ego(rgb)), msgs2, mask)
    assert torch.allclose(ref, out, atol=1e-5)


def test_fully_masked_row_pools_to_zeros_with_finite_grads():
    r = make_receiver()
    nn.init.normal_(r.v.weight)
    rgb, msgs, mask = rand_inputs()
    mask.zero_()
    msgs.requires_grad_(True)
    pooled = r.pool_neighbors(r.ego_proj(r.encode_ego(rgb)), msgs, mask)
    assert pooled.abs().max() < 1e-6
    pooled.sum().backward()
    assert torch.isfinite(msgs.grad).all()


def test_permutation_invariance_over_slots():
    r = make_receiver()
    nn.init.normal_(r.v.weight)
    rgb, msgs, mask = rand_inputs()
    perm = torch.tensor([2, 0, 1])
    ego = r.ego_proj(r.encode_ego(rgb))
    a = r.pool_neighbors(ego, msgs, mask)
    b = r.pool_neighbors(ego, msgs[:, perm], mask[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_none_floor_has_no_message_parameters():
    r = make_receiver(broadcast_dim=0)
    assert r.msg_proj is None
    rgb = torch.rand(B, 3, 64, 64)
    logits, value = r(rgb, torch.zeros(B, 0, 1), torch.zeros(B, 0))
    assert logits.shape == (B, 3) and value.shape == (B, 1)


def test_gaussian_policy_api_shapes():
    r = make_receiver()
    rgb, msgs, mask = rand_inputs()
    action, logp, ent, value = r.get_action_and_value(rgb, msgs, mask)
    assert action.shape == (B, 3)
    assert logp.shape == (B,)
    assert ent.shape == (B,)
    assert value.shape == (B, 1)
    # log-prob of a provided action is reproducible
    _, logp2, _, _ = r.get_action_and_value(rgb, msgs, mask, action=action)
    assert torch.allclose(logp, logp2, atol=1e-6)
