#!/usr/bin/env bash
# Round 3: continue the round-2 winner. r2A (rescue recipe, 6M) ended with
# canonical success climbing 0.07/0.14 -> 0.27/0.34 over the last 130 iters
# and the LR annealed to zero mid-climb; greedy no-jitter eval reads
# obedience 0.840 / success 0.504. Fresh LR schedule, same stable recipe,
# three variants:
#   cont1  verbatim continuation        the straight bet
#   cont2  ent 0.01                     both modes are established; lower
#                                       entropy may let success consolidate
#   cont3  jitter 0.20, ent 0.01        keeps placement-error robustness while
#                                       easing the hardest spawns
#
#   bash spike/launch_realcam_fix3.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
JEPA=checkpoints/jepa_realcam20.pt
OUT=runs/realcam20
INIT="$OUT/r2A_rescue6M.pt"

run () {  # name gpu seed extra...
  local name="$1" gpu="$2" seed="$3"; shift 3
  if [ -f "$OUT/$name.pt" ]; then echo "=== $name: exists, skipping ==="; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u rl/train_route_obey.py \
    --init_nav "$INIT" \
    --route_abort_wrong 1 \
    --rew_wrong_corridor 0.0 \
    --route_shaping 1 \
    --explore_window 0 \
    --total_timesteps 6000000 \
    --spawn_route 0.5 --spawn_route_anneal 0.6 \
    --seed "$seed" \
    --jepa_ckpt "$JEPA" \
    --log_csv "$OUT/$name.csv" \
    --run_json "$OUT/$name.json" \
    --save "$OUT/$name.pt" \
    "$@" \
    > "$OUT/$name.log" 2>&1 \
    && echo "${name}_DONE" || echo "${name}_FAILED"
}

run r3_cont1 1 11 --spawn_yaw_jitter 0.35 --ent_coef 0.02 &
run r3_cont2 2 12 --spawn_yaw_jitter 0.35 --ent_coef 0.01 &
run r3_cont3 3 13 --spawn_yaw_jitter 0.20 --ent_coef 0.01 &
wait
echo "ROUND3_ALL_DONE"
