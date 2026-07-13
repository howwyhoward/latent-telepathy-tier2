"""JEPA encoder + action-conditioned predictor for Tier 2 Stage A (pixels).

Faithful port of Tier 1's models/jepa.py (BYOL / I-JEPA recipe: online
encoder, EMA target encoder, stop-gradient, normalized-MSE latent loss,
VICReg variance/covariance safety net). Differences forced by the domain:

  - `PixelEncoder`: 64x64x3 RGB -> z in R^64 (Tier 1: one-hot grid patch -> 32)
  - `Predictor` conditions on continuous cmd_vel (vx, vy, wz) via a small MLP
    (Tier 1: nn.Embedding over 5 discrete actions)
  - `SegDecoder`: z -> 16x16 semantic-class logits (Tier 1: per-cell logits
    over the patch). Targets come from the sim's ground-truth segmentation,
    downsampled — the same "reconstruction is spatial and class-weighted"
    trick that fixed Tier 1's goal-blindness, with hazard/goal/agent
    upweighted. The decoder is a training aid and is discarded at freeze.

Isaac-free: trains from an .npz dataset on disk, unit-testable without Kit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import LATENT_DIM, N_ACTIONS

# Semantic classes for the reconstruction target, in a fixed order.
# `idToLabels` from the sim is remapped to these indices at collection time.
SEG_CLASSES = ("background", "wall", "hazard", "goal", "agent")
SEG_UPWEIGHTED = ("hazard", "goal", "agent")  # rare but thesis-critical
SEG_RES = 16  # decoder output resolution (64 -> 16 keeps WHERE, cheap to decode)


class PixelEncoder(nn.Module):
    """Small CNN: (B, 3, 64, 64) float in [0, 1] -> (B, latent_dim).

    Same downsampling ladder as the M7 agent trunk (8/4, 4/2, 3/1) so the
    receptive-field behavior that provably supports control transfers, but
    with an independent, narrower head sized for a broadcastable latent.
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 3, 64, 64)).flatten(1).shape[1]
        self.head = nn.Linear(flat, latent_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(rgb).flatten(1))


class SegDecoder(nn.Module):
    """z -> (B, n_classes, SEG_RES, SEG_RES) semantic logits.

    Spatial per-pixel decoding (not "is a hazard visible" pooling): forces z
    to encode WHERE the hazard/goal/agent sit, which is the bearing content a
    neighbor needs. Mirrors Tier 1's PatchDecoder.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_classes: int = len(SEG_CLASSES)):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),  # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(-1, 64, 4, 4))


def reconstruction_loss(
    logits: torch.Tensor, target_seg: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    """Class-weighted per-pixel cross-entropy (target: (B, SEG_RES, SEG_RES) ints)."""
    return F.cross_entropy(logits, target_seg.long(), weight=class_weights)


def make_class_weights(upweight: float = 10.0, device="cpu") -> torch.Tensor:
    """Tier 1's recipe: rare thesis-critical classes get `upweight`, rest 1."""
    w = torch.ones(len(SEG_CLASSES), device=device)
    for name in SEG_UPWEIGHTED:
        w[SEG_CLASSES.index(name)] = upweight
    return w


class Predictor(nn.Module):
    """Action-conditioned latent dynamics: (z_t, a_t) -> z_hat_{t+1}.

    a_t is the continuous normalized cmd_vel (vx, vy, wz) in [-1, 1]; a small
    MLP replaces Tier 1's discrete-action embedding.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_actions: int = N_ACTIONS,
        action_emb: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        self.action_emb = nn.Sequential(
            nn.Linear(n_actions, action_emb),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.action_emb(action)], dim=-1))


class JEPA(nn.Module):
    """Online encoder + EMA target encoder + predictor (Tier 1 structure)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_actions: int = N_ACTIONS,
        action_emb: int = 16,
        predictor_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = PixelEncoder(latent_dim)
        self.target_encoder = PixelEncoder(latent_dim)
        self.predictor = Predictor(latent_dim, n_actions, action_emb, predictor_hidden)

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, tau: float = 0.99) -> None:
        """EMA step: target <- tau*target + (1-tau)*online."""
        for tp, op in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.mul_(tau).add_(op, alpha=1.0 - tau)
        for tb, ob in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            tb.copy_(ob)

    def forward(
        self,
        rgb_t: torch.Tensor,
        action_t: torch.Tensor,
        rgb_tp1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = self.encoder(rgb_t)
        z_pred = self.predictor(z_t, action_t)
        with torch.no_grad():
            z_target = self.target_encoder(rgb_tp1)
        return z_pred, z_target, z_t


def jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """BYOL-style normalized MSE (= 2 - 2*cosine); target is detached."""
    p = F.normalize(z_pred, dim=-1)
    t = F.normalize(z_target.detach(), dim=-1)
    return ((p - t) ** 2).sum(dim=-1).mean()


# -- VICReg anti-collapse terms, verbatim from Tier 1 ----------------------


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge keeping each dimension's std at or above `gamma`."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalize squared off-diagonal covariances so dimensions decorrelate."""
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return (off_diag**2).sum() / d


def latent_stats(z: torch.Tensor) -> tuple[float, float, float]:
    """(mean per-dim std, min per-dim std, effective rank) — collapse monitor."""
    z = z.detach().float().cpu()
    std = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc)
    p = sv / (sv.sum() + 1e-12)
    eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return std.mean().item(), std.min().item(), eff_rank
