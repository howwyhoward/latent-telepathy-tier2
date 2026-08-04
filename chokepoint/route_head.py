"""The race v8 route head — the entire trainable surface of stage 2.

message -> 2 logits, one categorical decision per episode, credited with the
episode's return (a contextual bandit). Race v1-v7 showed why this must be a
discrete first-class decision: a corridor preference living in the lateral
component of a Gaussian action stream pays a 1/sigma^2 gradient penalty for
its own exploration. Sampling a bit does not.

Lives in chokepoint/ (not rl/) because it is part of the deployed
architecture: on the RoboMasters this module is what turns the received
latent into a route command for the executor.
"""

import torch
import torch.nn as nn


class RouteHead(nn.Module):
    def __init__(self, wire: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(wire, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, 2)
        self.value = nn.Linear(hidden, 1)
        # start exactly uniform: no architectural prior on either corridor
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, msg: torch.Tensor):
        h = self.trunk(msg)
        return self.logits(h), self.value(h).squeeze(-1)
