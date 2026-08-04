"""Attention-fusion receiver policy, ported from Tier 1 (models/receiver.py).

The masked scaled-dot-product pooling, the frozen-encoder discipline, and the
zero-init value path (the M10c exploration fix: every condition STARTS as the
`none` policy and recruits its channel only as gradients prove it useful)
transfer verbatim — they operate on latents and masks, which are
modality-agnostic.

What changed for Isaac:

  - Ego perception consumes (B, 3, H, W) RGB instead of one-hot patches; the
    encoder is whatever frozen module the JEPA phase produces (Phase 2).
  - The action head is a diagonal Gaussian over the 3-dim cmd_vel instead of
    a Categorical over 5 grid actions; log-prob/entropy sum over action dims.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .constants import N_ACTIONS


class AttentionReceiver(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        broadcast_dim: int,
        latent_dim: int = 64,
        d_model: int = 128,
        n_actions: int = N_ACTIONS,
        ego_adapter: bool = False,
        route_dim: int = 0,
    ):
        super().__init__()
        # Frozen shared perception: no gradients, eval mode (no BN/dropout drift).
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.broadcast_dim = broadcast_dim
        self.d_model = d_model

        # Ego pathway (identical across all conditions).
        self.ego_proj = nn.Linear(latent_dim, d_model)
        # Optional ego-only adapter: shapes absolute level if the frozen latent
        # is not control-optimal. Identical across conditions, never touches
        # the contrast.
        self.adapter = (
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
            if ego_adapter
            else None
        )

        # Neighbor channel. broadcast_dim == 0 is the "none" floor: no channel,
        # so there is no message projection and pooling returns zeros.
        if broadcast_dim > 0:
            self.msg_proj = nn.Linear(broadcast_dim, d_model)
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            # Zero-init the value path (see Tier 1 §8.5): at init the pooled
            # neighbor context is exactly zero, so every message condition
            # starts as the `none` policy. Without this, randomly-projected
            # messages act as a high-variance distractor that stalls PPO on
            # sparse-reward tasks (observed in Tier 1: raw and z_t flat at
            # chance while `none` learned).
            nn.init.zeros_(self.v.weight)
            nn.init.zeros_(self.v.bias)
        else:
            self.msg_proj = None

        # Route command (stage 1.5+): a one-hot the low-level controller is
        # trained to obey, appended to the feature vector rather than mixed into
        # the ego embedding so a trunk trained without it loads unchanged.
        self.route_dim = route_dim
        feat_dim = 2 * d_model + route_dim

        self.actor = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_actions),
        )
        # State-independent log-std, PPO-standard for continuous control.
        self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    # -- perception ---------------------------------------------------------

    def encode_ego(self, ego_rgb: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder embedding of the ego camera frame. No grad into E."""
        with torch.no_grad():
            z = self.encoder(ego_rgb)
        return z

    # -- masked attention pooling (verbatim from Tier 1) ---------------------

    def pool_neighbors(
        self, ego_embed: torch.Tensor, messages: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked scaled-dot-product attention: ego query over neighbor messages.

        messages: (B, N, broadcast_dim) ; mask: (B, N) with 1 = populated.
        Returns pooled neighbor context (B, d_model). An all-masked row pools
        to zeros (the agent simply has no neighbors this step).
        """
        if self.msg_proj is None or messages.shape[1] == 0:
            return torch.zeros_like(ego_embed)

        m = self.msg_proj(messages)            # (B, N, d_model)
        q = self.q(ego_embed).unsqueeze(1)     # (B, 1, d_model)
        k = self.k(m)                          # (B, N, d_model)
        v = self.v(m)                          # (B, N, d_model)

        scores = (q * k).sum(-1) / math.sqrt(self.d_model)  # (B, N)
        # Masked softmax done by hand: exponentiate, then ZERO masked slots by
        # multiplying with the mask (never an all-`-1e9` row, which yields a
        # uniform distribution whose backward is NaN). A fully-masked row gets
        # denom -> 1e-9, so attn -> 0 and the agent pools to zeros, with clean
        # finite gradients. Max-subtraction (detached) keeps exp() stable.
        mask_f = mask.float()
        scores = scores - scores.max(dim=1, keepdim=True).values.detach()
        weights = torch.exp(scores) * mask_f                # (B, N)
        attn = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (attn.unsqueeze(-1) * v).sum(dim=1)        # (B, d_model)
        return pooled

    # -- forward / policy API ----------------------------------------------

    def features(
        self,
        ego_rgb: torch.Tensor,
        messages: torch.Tensor,
        mask: torch.Tensor,
        route: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encode_ego(ego_rgb)
        if self.adapter is not None:
            z = z + self.adapter(z)
        ego_embed = self.ego_proj(z)
        pooled = self.pool_neighbors(ego_embed, messages, mask)
        parts = [ego_embed, pooled]
        if self.route_dim > 0:
            assert route is not None, "route_dim > 0 but no route command given"
            parts.append(route)
        return torch.cat(parts, dim=-1)

    def forward(self, ego_rgb, messages, mask, route=None):
        h = self.features(ego_rgb, messages, mask, route)
        return self.actor(h), self.critic(h)

    def get_value(self, ego_rgb, messages, mask, route=None):
        return self.critic(self.features(ego_rgb, messages, mask, route))

    def get_action_and_value(self, ego_rgb, messages, mask, action=None, route=None):
        h = self.features(ego_rgb, messages, mask, route)
        mean = self.actor(h)
        dist = Normal(mean, self.log_std.exp())
        if action is None:
            action = dist.sample()
        # sum over action dims: one log-prob/entropy scalar per env
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(h),
        )

    # -- warm start ---------------------------------------------------------

    def load_trunk(self, src: dict) -> dict:
        """Load a shared trunk, tolerating a narrower source feature vector.

        Message-branch keys absent from a `none` source keep their zero-init
        value path, so every condition still starts as the `none` policy. When
        this model adds a route command, actor/critic first layers are wider
        than the source's: the shared columns are copied and the route columns
        left at zero, so the loaded controller is initially route-BLIND and
        behaves exactly as its source did. Silently skipping those layers on a
        shape mismatch would instead discard the whole trunk.
        """
        own = self.state_dict()
        loaded, widened = [], []
        for k, v in src.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            elif own[k].dim() == 2 and own[k].shape[0] == v.shape[0] \
                    and own[k].shape[1] > v.shape[1]:
                own[k] = torch.zeros_like(own[k])
                own[k][:, : v.shape[1]] = v
                loaded.append(k)
                widened.append(k)
        self.load_state_dict(own)
        return {"loaded": loaded, "widened": widened, "total": len(own)}
