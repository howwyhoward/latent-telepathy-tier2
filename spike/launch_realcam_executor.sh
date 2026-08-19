#!/usr/bin/env bash
# Executor retrain against the REAL-CAMERA encoder (checkpoints/jepa_realcam.pt,
# 15 Aug handoff geometry: 0.18 m, 2.1 deg pitch, 32 deg HFOV).
#
# Replays the proven chain with the new latent — warm starts from old-encoder
# checkpoints are invalid because ego_proj consumed a different latent space:
#   1. nav pretrain     (train_race.py, condition none, mouth spawns)  ~3M steps
#   2. route obey v4    (abort_mouth recipe, init from 1)              ~3M steps
#   3. route obey cont  (same recipe continued, init from 2)           ~3M steps
#
#   bash spike/launch_realcam_executor.sh <gpu> [jepa_ckpt] [out_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
GPU="${1:-0}"
JEPA="${2:-checkpoints/jepa_realcam20.pt}"
OUT="${3:-runs/realcam20}"
mkdir -p "$OUT"

# stages are skipped if their checkpoint already exists, so the script can be
# relaunched after an interruption without redoing finished work
if [ -f "$OUT/nav_s1_mouth.pt" ]; then
  echo "=== STAGE 1: checkpoint exists, skipping ==="
else
echo "=== STAGE 1: nav pretrain (mouth spawns) ==="
CUDA_VISIBLE_DEVICES="$GPU" python -u rl/train_race.py \
  --condition none \
  --spawn_curriculum 0.0 \
  --spawn_mouths 0.5 \
  --seed 1 \
  --jepa_ckpt "$JEPA" \
  --log_csv "$OUT/nav_s1_mouth.csv" \
  --run_json "$OUT/nav_s1_mouth.json" \
  --save "$OUT/nav_s1_mouth.pt" \
  > "$OUT/nav_s1_mouth.log" 2>&1 || { echo "STAGE1_FAILED"; exit 1; }
echo "STAGE1_DONE"
fi

if [ -f "$OUT/route_v4.pt" ]; then
  echo "=== STAGE 2: checkpoint exists, skipping ==="
else
echo "=== STAGE 2: route obey, abort_mouth recipe ==="
CUDA_VISIBLE_DEVICES="$GPU" python -u rl/train_route_obey.py \
  --init_nav "$OUT/nav_s1_mouth.pt" \
  --route_abort_wrong 1 \
  --rew_wrong_corridor 0.0 \
  --route_shaping 1 \
  --spawn_mouths 0.5 \
  --spawn_yaw_jitter 0.0 \
  --explore_window 0 \
  --seed 2 \
  --jepa_ckpt "$JEPA" \
  --log_csv "$OUT/route_v4.csv" \
  --run_json "$OUT/route_v4.json" \
  --save "$OUT/route_v4.pt" \
  > "$OUT/route_v4.log" 2>&1 || { echo "STAGE2_FAILED"; exit 1; }
echo "STAGE2_DONE"
fi

echo "=== STAGE 3: continuation to the gate ==="
CUDA_VISIBLE_DEVICES="$GPU" python -u rl/train_route_obey.py \
  --init_nav "$OUT/route_v4.pt" \
  --route_abort_wrong 1 \
  --rew_wrong_corridor 0.0 \
  --route_shaping 1 \
  --spawn_mouths 0.5 \
  --spawn_yaw_jitter 0.0 \
  --explore_window 0 \
  --seed 3 \
  --jepa_ckpt "$JEPA" \
  --log_csv "$OUT/route_cont.csv" \
  --run_json "$OUT/route_cont.json" \
  --save "$OUT/route_cont.pt" \
  > "$OUT/route_cont.log" 2>&1 || { echo "STAGE3_FAILED"; exit 1; }
echo "STAGE3_DONE"
echo "ALL_STAGES_DONE"
