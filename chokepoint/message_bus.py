"""Communication seam, ported from Tier 1 (envs/message_bus.py).

Same architecture, same experimental logic: the message conditions
(none / position / kinematic / z_t / z_hat / raw obs) are each just a
different override of `get_broadcast_content`; nothing else in the stack
changes between them. The anchored-delivery redesign (every slot prefixed
with the sender's position RELATIVE to the receiver, added at delivery time,
identically for every condition) carries over unchanged — it is the reason
the race question stays "what does content add BEYOND position-sharing".

What changed for Isaac (and why):

  - Vectorized over parallel envs: everything is a torch tensor on the env
    device. Shapes grow a leading (num_envs,) dim vs the Tier 1 NumPy bus.
  - Distance metric: Euclidean meters instead of Chebyshev cells (the grid
    metric has no privileged 3D analog). Default comm_radius 3.0 m = Tier 1's
    6 cells x 0.5 m/cell.
  - `sim` is now the ChokepointEnv; positions come from PhysX root states in
    env-local coordinates.
  - Agents are addressed by name (AGENT_NAMES order fixes slot identity), and
    all agents are always live (nobody dies in the chokepoint race).
"""

import torch

from .constants import AGENT_NAMES


def _pad(vec: torch.Tensor, n: int) -> torch.Tensor:
    """Fit (B, k) content into the matched channel width `n` (zero-padded).

    Padding (e.g. position's 2 floats into a 64-slot channel) is the honest
    bandwidth control: equal floats on the wire, less to say.
    """
    b, k = vec.shape
    if k == n:
        return vec
    out = torch.zeros(b, n, dtype=torch.float32, device=vec.device)
    out[:, : min(k, n)] = vec[:, : min(k, n)]
    return out


class MessageBus:
    """Delivers fixed-shape neighbor messages each control step.

    Neighbors are other agents within `comm_radius` Euclidean meters.
    Output per agent:
      messages : (num_envs, n_agents - 1, wire_dim) float32, zero for
                 out-of-range slots
      mask     : (num_envs, n_agents - 1)           int8, 1 where populated

    Slot identity is stable: slot order is AGENT_NAMES with ego skipped,
    independent of who is currently in range — the mask says whether a slot
    is populated this step.
    """

    def __init__(
        self,
        comm_radius: float = 3.0,
        broadcast_dim: int = 8,
        anchored: bool = False,
    ):
        self.agents = list(AGENT_NAMES)
        self.comm_radius = comm_radius
        self.broadcast_dim = broadcast_dim
        # Anchor = sender position relative to receiver, normalized by
        # comm_radius (in-range anchors lie in [-1, 1]). See Tier 1 §8.4.
        self.anchored = anchored

    @property
    def wire_dim(self) -> int:
        """Width of a delivered slot: anchor prefix (if any) + content."""
        return self.broadcast_dim + (2 if self.anchored else 0)

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        """What `agent` broadcasts this step, (num_envs, broadcast_dim).

        Default: silence (the `none` floor). Override per condition.
        """
        return torch.zeros(env.num_envs, self.broadcast_dim, device=env.device)

    def deliver(self, env) -> dict:
        """Compute messages + mask for every agent. {agent: (messages, mask)}."""
        pos = {a: env._local_pos(a) for a in self.agents}
        content = {a: self.get_broadcast_content(a, env) for a in self.agents}

        result = {}
        for ego in self.agents:
            slots = [a for a in self.agents if a != ego]
            messages = torch.zeros(
                env.num_envs, len(slots), self.wire_dim, device=env.device
            )
            mask = torch.zeros(
                env.num_envs, len(slots), dtype=torch.int8, device=env.device
            )
            for si, other in enumerate(slots):
                delta = pos[other] - pos[ego]                      # (B, 2)
                in_range = delta.norm(dim=1) <= self.comm_radius   # (B,)
                if self.anchored:
                    messages[:, si, 0:2] = delta / self.comm_radius
                    messages[:, si, 2:] = content[other]
                else:
                    messages[:, si] = content[other]
                # out-of-range slots carry nothing, matching Tier 1 semantics
                messages[:, si][~in_range] = 0.0
                mask[:, si] = in_range.to(torch.int8)
            result[ego] = (messages, mask)
        return result


class PositionBroadcast(MessageBus):
    """`position` condition: normalized (x, y). broadcast_dim fixed at 2."""

    def __init__(self, comm_radius: float = 3.0, anchored: bool = False):
        super().__init__(comm_radius=comm_radius, broadcast_dim=2, anchored=anchored)

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        extent = env.cfg.cell * self._grid_size(env)
        return (env._local_pos(agent) + extent / 2) / extent  # ~[0, 1]

    @staticmethod
    def _grid_size(env) -> int:
        return env._geo.size


class KinematicBroadcast(MessageBus):
    """Kinematic steelman: normalized position + constant-velocity trajectory.

    Tier 1 extrapolated the policy's grid action; the continuous analog
    extrapolates the robot's CURRENT world velocity over `horizon` control
    steps — position + short predicted trajectory, same steelman.
    """

    def __init__(
        self,
        *,
        horizon: int = 3,
        comm_radius: float = 3.0,
        broadcast_dim: int = 64,
        anchored: bool = False,
    ):
        super().__init__(
            comm_radius=comm_radius, broadcast_dim=broadcast_dim, anchored=anchored
        )
        self.horizon = horizon

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        extent = env.cfg.cell * env._geo.size
        dt = env.cfg.sim.dt * env.cfg.decimation
        p = env._local_pos(agent)
        v = env.scene[agent].data.root_lin_vel_w[:, :2]
        coords = [(p + extent / 2) / extent]
        q = p
        for _ in range(self.horizon):
            q = q + v * dt
            coords.append((q + extent / 2) / extent)
        return _pad(torch.cat(coords, dim=1), self.broadcast_dim)


class LatentBroadcast(MessageBus):
    """Condition C1: the frozen encoder's latent z_t of the agent's own camera
    view, fitted to the matched channel width. Encoder frozen, eval, no grad.
    """

    def __init__(
        self,
        encoder,
        *,
        comm_radius: float = 3.0,
        broadcast_dim: int = 64,
        anchored: bool = False,
    ):
        super().__init__(
            comm_radius=comm_radius, broadcast_dim=broadcast_dim, anchored=anchored
        )
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def _ego_rgb(self, agent: str, env) -> torch.Tensor:
        # (B, H, W, 3) in [0,1] -> (B, 3, H, W), the conv layout
        rgb = env.scene[f"cam_{agent}"].data.output["rgb"].float() / 255.0
        return rgb.permute(0, 3, 1, 2)

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(self._ego_rgb(agent, env))
        return _pad(z, self.broadcast_dim)


class PredictedLatentBroadcast(LatentBroadcast):
    """Condition C2: broadcast z_hat = P(z_t, a).

    `action_source(agent, env) -> (B, 3) cmd_vel tensor`; default is the
    zero command (Tier 1's STAY analog). The race harness wires the policy's
    chosen action so the message is the anticipated next view under intent.
    """

    def __init__(
        self,
        encoder,
        predictor,
        *,
        action_source=None,
        comm_radius: float = 3.0,
        broadcast_dim: int = 64,
        anchored: bool = False,
    ):
        super().__init__(
            encoder,
            comm_radius=comm_radius,
            broadcast_dim=broadcast_dim,
            anchored=anchored,
        )
        self.predictor = predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.action_source = action_source or (
            lambda agent, env: torch.zeros(env.num_envs, 3, device=env.device)
        )

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(self._ego_rgb(agent, env))
            zhat = self.predictor(z, self.action_source(agent, env))
        return _pad(zhat, self.broadcast_dim)


class RawObsBroadcast(MessageBus):
    """Ceiling condition: the agent's full camera frame, flattened.

    Natural width H*W*3 (12288 at 64x64) — the unconstrained 'share
    everything' upper bound, deliberately NOT matched to the 64-D channel,
    and precisely the message that cannot fit the real network.
    """

    def __init__(
        self,
        *,
        resolution: int = 64,
        comm_radius: float = 3.0,
        anchored: bool = False,
    ):
        super().__init__(
            comm_radius=comm_radius,
            broadcast_dim=resolution * resolution * 3,
            anchored=anchored,
        )

    def get_broadcast_content(self, agent: str, env) -> torch.Tensor:
        rgb = env.scene[f"cam_{agent}"].data.output["rgb"].float() / 255.0
        return rgb.reshape(env.num_envs, -1)
