#!/usr/bin/env bash
# Round 4 (final): consolidation from the balanced r2A checkpoint.
#
# Round-3 continuations of the rescue recipe traded success for obedience
# (greedy: r2A 0.840/0.504 -> cont3 0.887/0.340), so more curriculum now hurts
# the canonical-start skill. The original chain's gate-passing step was a plain
# v4-recipe continuation (mouth spawns, no jitter) from a BALANCED checkpoint;
# r2C_consol's collapse started from an already-collapsed init and proves
# nothing about this case. Two seeds; keep whichever beats r2A on greedy
# success, else r2A stays the export.
#
#   bash spike/launch_realcam_fix4.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
JEPA=checkpoints/jepa_realcam20.pt
OUT=runs/realcam20
INIT="$OUT/r2A_rescue6M.pt"

run () {  # name gpu seed
  local name="$1" gpu="$2" seed="$3"
  if [ -f "$OUT/$name.pt" ]; then echo "=== $name: exists, skipping ==="; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u rl/train_route_obey.py \
    --init_nav "$INIT" \
    --route_abort_wrong 1 \
    --rew_wrong_corridor 0.0 \
    --route_shaping 1 \
    --explore_window 0 \
    --spawn_mouths 0.5 --spawn_yaw_jitter 0.0 --ent_coef 0.01 \
    --seed "$seed" \
    --jepa_ckpt "$JEPA" \
    --log_csv "$OUT/$name.csv" \
    --run_json "$OUT/$name.json" \
    --save "$OUT/$name.pt" \
    > "$OUT/$name.log" 2>&1 \
    && echo "${name}_DONE" || echo "${name}_FAILED"
}

run r4_consolA 1 14 &
run r4_consolB 2 15 &
wait
echo "ROUND4_ALL_DONE"
