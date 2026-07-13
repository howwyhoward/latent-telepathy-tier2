#!/usr/bin/env bash
# Activate the Tier 2 Isaac environment. Usage:  source setup/env.sh
#
# Everything heavy lives on /data because / is 100% full (13 GB free).
# ~/.cache/ov, ~/.local/share/ov, ~/.cache/warp, ~/.nvidia-omniverse are
# symlinks into /data/howard/isaac/ — do not remove them.

export ISAAC_ROOT=/data/howard/isaac
export PIP_CACHE_DIR=$ISAAC_ROOT/cache/pip
export TMPDIR=$ISAAC_ROOT/tmp

# GPUs 2 and 3 run labmates' jobs; GPU 0 reports a fan sensor error.
export CUDA_VISIBLE_DEVICES=1

# Isaac Sim asset/shader caches (belt and suspenders on top of the symlinks)
export OMNI_USER_CACHE_DIR=$ISAAC_ROOT/cache/ov

# Accept the NVIDIA EULA non-interactively (required for headless first run)
export OMNI_KIT_ACCEPT_EULA=YES

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate $ISAAC_ROOT/envs/isaaclab

echo "isaaclab env active: $(python --version) | GPU=$CUDA_VISIBLE_DEVICES | caches on /data"
