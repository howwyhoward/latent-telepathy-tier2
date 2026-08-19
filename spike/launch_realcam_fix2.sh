#!/usr/bin/env bash
# Round 2. Round-1 evidence (all three died at ~82% in a machine outage, logs
# survived): the commanded-route curriculum rescue (fixC) broke the top-corridor
# collapse -- canonical obedience 0.74/0.78 balanced and climbing -- but success
# lagged (0.14/0.01) at the 3M cutoff. The re-roll and from-scratch curriculum
# both re-collapsed, so the rescue is the only recipe that works; it needs room.
#
#   A  fixC verbatim, 6M steps        the straight bet: let obedience convert
#                                     into success.
#   B  fixC with yaw jitter 0.15, 6M  tests whether 0.35 rad jitter was what
#                                     kept success down.
#   C  fixC 3M -> consolidation 3M    restore balance, then the original cont
#                                     recipe (mouth spawns, no jitter) to
#                                     consolidate traversal into success.
#
#   bash spike/launch_realcam_fix2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
JEPA=checkpoints/jepa_realcam20.pt
OUT=runs/realcam20
INIT="$OUT/route_cont.pt"
mkdir -p "$OUT"

run () {  # name gpu init seed extra...
  local name="$1" gpu="$2" init="$3" seed="$4"; shift 4
  if [ -f "$OUT/$name.pt" ]; then echo "=== $name: exists, skipping ==="; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u rl/train_route_obey.py \
    --init_nav "$init" \
    --route_abort_wrong 1 \
    --rew_wrong_corridor 0.0 \
    --route_shaping 1 \
    --explore_window 0 \
    --seed "$seed" \
    --jepa_ckpt "$JEPA" \
    --log_csv "$OUT/$name.csv" \
    --run_json "$OUT/$name.json" \
    --save "$OUT/$name.pt" \
    "$@" \
    > "$OUT/$name.log" 2>&1 \
    && echo "${name}_DONE" || { echo "${name}_FAILED"; return 1; }
}

run r2A_rescue6M 1 "$INIT" 7 \
    --total_timesteps 6000000 \
    --spawn_route 0.5 --spawn_route_anneal 0.6 --spawn_yaw_jitter 0.35 --ent_coef 0.02 &

run r2B_lowjit6M 2 "$INIT" 8 \
    --total_timesteps 6000000 \
    --spawn_route 0.5 --spawn_route_anneal 0.6 --spawn_yaw_jitter 0.15 --ent_coef 0.02 &

(
  run r2C_rescue 3 "$INIT" 9 \
      --spawn_route 0.5 --spawn_route_anneal 0.6 --spawn_yaw_jitter 0.35 --ent_coef 0.02 \
  && run r2C_consol 3 "$OUT/r2C_rescue.pt" 10 \
      --spawn_mouths 0.5 --spawn_yaw_jitter 0.0
) &
wait
echo "ROUND2_ALL_DONE"
