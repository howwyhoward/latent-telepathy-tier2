"""Tier 2 chokepoint: Isaac Lab port of the Tier 1 decisive scenario (M10c)."""

import torch

# Driver 550 predates the CUDA 12.8 APIs that this torch build's cuDNN 9 needs
# at init (CUDNN_STATUS_NOT_INITIALIZED on the first conv). Plain CUDA kernels
# work fine via minor-version compatibility, so fall back to them; at 64x64
# with a small encoder the speed difference is negligible. Remove when the
# server driver reaches >= 570.
torch.backends.cudnn.enabled = False
