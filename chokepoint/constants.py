"""Single source of truth for agent identity and shared dims (Tier 1 pattern).

Kept Isaac-free so the message bus and receiver can be unit-tested without
launching Kit.
"""

# Slot identity in the message bus follows this order (ego skipped).
AGENT_NAMES = ("navigator", "scout")

# cmd_vel action space: (vx, vy, wz), normalized to [-1, 1].
N_ACTIONS = 3

# Frozen-encoder latent width (matches Tier 1's 64-D channel).
LATENT_DIM = 64
