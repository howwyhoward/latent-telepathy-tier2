"""End-to-end pixel PPO agent for the M7 positive control.

Lives in `chokepoint` (not `rl/`) because Tier 1's repo is on sys.path and
also has an `rl` package; this package name is unique to Tier 2. Import-safe:
no Kit, no argparse.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .constants import N_ACTIONS

LEARNER = "navigator"


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Nature-CNN-small trunk shared by actor and critic; separate heads.

    Small-std actor init + log_std=-0.5 keeps the initial policy a gentle
    zero-mean wander, the continuous analogue of near-uniform Categorical.
    """

    def __init__(self, n_actions: int = N_ACTIONS, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, hidden)),
            nn.ReLU(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.trunk(x))

    def get_action_and_value(self, x, action=None):
        h = self.trunk(x)
        mean = self.actor_mean(h)
        dist = Normal(mean, self.log_std.exp())
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(h),
        )
