#!/usr/bin/env bash
# Round-1 fixes for the realcam20 route-obedience collapse.
#
# Failure being addressed: the v4-recipe replication under the 32-deg camera
# was balanced through ~iter 250 (canonical obey 0.61/0.52) and then mode-
# collapsed onto the top corridor (final 0.87/0.07); the continuation stayed
# collapsed for all 3M steps. Under the narrow FOV both mouths are out of view
# at the canonical heading, so "always top" is a strong local optimum once
# entropy anneals.
#
# Three parallel counters, one per free GPU:
#   A re-roll   v4 recipe from the nav trunk, new seed, ent_coef 0.02 --
#               tests whether the collapse is seed luck plus premature
#               entropy decay.
#   B curric    v5 reverse curriculum along the COMMANDED route (guarantees
#               balanced successes on both corridors, the direct antidote to
#               one-sided collapse) + 0.35 rad yaw jitter, which bridges the
#               heading gap that made the original v5 backfire.
#   C rescue    continuation from the collapsed route_cont.pt with the same
#               curriculum + jitter + ent 0.02: commanded-route spawns feed
#               the bottom corridor successes the collapsed policy never sees.
#
#   bash spike/launch_realcam_fix.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
JEPA=checkpoints/jepa_realcam20.pt
OUT=runs/realcam20
mkdir -p "$OUT"

run () {  # name gpu init seed extra...
  local name="$1" gpu="$2" init="$3" seed="$4"; shift 4
  if [ -f "$OUT/$name.pt" ]; then echo "=== $name: exists, skipping ==="; return; fi
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
    && echo "${name}_DONE" || echo "${name}_FAILED"
}

run fixA_reroll 1 "$OUT/nav_s1_mouth.pt" 4 \
    --spawn_mouths 0.5 --spawn_yaw_jitter 0.0 --ent_coef 0.02 &
run fixB_curric 2 "$OUT/nav_s1_mouth.pt" 5 \
    --spawn_route 0.5 --spawn_route_anneal 0.6 --spawn_yaw_jitter 0.35 &
run fixC_rescue 3 "$OUT/route_cont.pt" 6 \
    --spawn_route 0.5 --spawn_route_anneal 0.6 --spawn_yaw_jitter 0.35 --ent_coef 0.02 &
wait
echo "ROUND1_ALL_DONE"
