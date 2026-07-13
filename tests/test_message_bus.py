"""Unit tests for the ported message bus — no Isaac required.

These re-establish the Tier 1 guarantees that the port must preserve:
slot identity, range masking, anchored delivery, honest zero-padding.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chokepoint.constants import AGENT_NAMES
from chokepoint.message_bus import (
    KinematicBroadcast,
    MessageBus,
    PositionBroadcast,
    _pad,
)


class FakeGeo:
    size = 20


class FakeCfg:
    cell = 0.5

    class sim:
        dt = 1 / 60

    decimation = 6


class FakeEnv:
    """Just enough of ChokepointEnv for the bus: positions + velocities."""

    def __init__(self, positions: dict, velocities: dict | None = None, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.cfg = FakeCfg()
        self._geo = FakeGeo()
        self._pos = {a: torch.as_tensor(p, dtype=torch.float32) for a, p in positions.items()}
        vel = velocities or {a: torch.zeros(num_envs, 2) for a in positions}
        self.scene = {
            a: type(
                "Robot",
                (),
                {"data": type("D", (), {"root_lin_vel_w": torch.cat([v, torch.zeros(num_envs, 1)], dim=1)})()},
            )()
            for a, v in vel.items()
        }

    def _local_pos(self, agent):
        return self._pos[agent]


def both_at(nav_xy, scout_xy, num_envs=4):
    return FakeEnv(
        {
            "navigator": torch.tensor(nav_xy).repeat(num_envs, 1),
            "scout": torch.tensor(scout_xy).repeat(num_envs, 1),
        },
        num_envs=num_envs,
    )


def test_pad_truncates_and_zero_fills():
    v = torch.ones(2, 3)
    out = _pad(v, 8)
    assert out.shape == (2, 8)
    assert out[:, :3].eq(1).all() and out[:, 3:].eq(0).all()
    assert _pad(torch.ones(2, 10), 4).shape == (2, 4)


def test_silence_floor_delivers_zero_content_but_valid_mask():
    env = both_at([0.0, 0.0], [1.0, 0.0])
    bus = MessageBus(comm_radius=3.0, broadcast_dim=8)
    out = bus.deliver(env)
    msgs, mask = out["navigator"]
    assert msgs.shape == (4, 1, 8)
    assert mask.eq(1).all()          # in range -> populated
    assert msgs.eq(0).all()          # ...but silent


def test_out_of_range_masks_slot_and_zeroes_message():
    env = both_at([0.0, 0.0], [10.0, 0.0])  # 10 m apart, radius 3
    bus = PositionBroadcast(comm_radius=3.0)
    out = bus.deliver(env)
    msgs, mask = out["navigator"]
    assert mask.eq(0).all()
    assert msgs.eq(0).all()


def test_anchor_is_relative_position_normalized():
    env = both_at([0.0, 0.0], [1.5, -0.6])
    bus = MessageBus(comm_radius=3.0, broadcast_dim=4, anchored=True)
    out = bus.deliver(env)
    msgs, mask = out["navigator"]
    assert bus.wire_dim == 6
    assert mask.eq(1).all()
    assert torch.allclose(msgs[:, 0, 0], torch.tensor(0.5))    # dx / radius
    assert torch.allclose(msgs[:, 0, 1], torch.tensor(-0.2))   # dy / radius
    # receiver-specific: the scout sees the opposite anchor
    msgs_s, _ = out["scout"]
    assert torch.allclose(msgs_s[:, 0, 0], torch.tensor(-0.5))


def test_anchor_identical_across_conditions():
    env = both_at([0.0, 0.0], [1.5, -0.6])
    anchors = []
    for bus in (
        MessageBus(comm_radius=3.0, broadcast_dim=8, anchored=True),
        PositionBroadcast(comm_radius=3.0, anchored=True),
        KinematicBroadcast(comm_radius=3.0, anchored=True),
    ):
        msgs, _ = bus.deliver(env)["navigator"]
        anchors.append(msgs[:, 0, :2])
    assert torch.allclose(anchors[0], anchors[1])
    assert torch.allclose(anchors[0], anchors[2])


def test_position_content_is_rank_2_padded():
    env = both_at([0.0, 0.0], [1.0, 1.0])
    bus = PositionBroadcast(comm_radius=3.0)
    msgs, _ = bus.deliver(env)["navigator"]
    assert msgs.shape[-1] == 2  # natural width; padding to 64 is the harness's call


def test_kinematic_extrapolates_current_velocity():
    n = 4
    env = FakeEnv(
        {
            "navigator": torch.zeros(n, 2),
            "scout": torch.tensor([1.0, 0.0]).repeat(n, 1),
        },
        velocities={
            "navigator": torch.zeros(n, 2),
            "scout": torch.tensor([0.5, 0.0]).repeat(n, 1),  # 0.5 m/s east
        },
        num_envs=n,
    )
    bus = KinematicBroadcast(horizon=2, comm_radius=3.0, broadcast_dim=64)
    msgs, _ = bus.deliver(env)["navigator"]
    extent = 0.5 * 20
    dt = (1 / 60) * 6
    # content: [p, p+v*dt, p+2v*dt] normalized; x coordinates strictly increasing
    x0 = msgs[0, 0, 0] * extent - extent / 2
    x1 = msgs[0, 0, 2] * extent - extent / 2
    x2 = msgs[0, 0, 4] * extent - extent / 2
    assert x1 == pytest.approx(x0 + 0.5 * dt, abs=1e-5)
    assert x2 == pytest.approx(x0 + 1.0 * dt, abs=1e-5)
    # padded tail is zero
    assert msgs[:, :, 6:].eq(0).all()


def test_slot_identity_matches_agent_names_order():
    # with 2 agents each ego has exactly 1 slot: the other agent
    env = both_at([0.0, 0.0], [1.0, 0.0])
    bus = MessageBus(comm_radius=3.0, broadcast_dim=2)
    out = bus.deliver(env)
    assert set(out.keys()) == set(AGENT_NAMES)
    for a in AGENT_NAMES:
        msgs, mask = out[a]
        assert msgs.shape[1] == len(AGENT_NAMES) - 1
