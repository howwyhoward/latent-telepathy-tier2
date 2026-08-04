#!/usr/bin/env bash
# Stage 1.5 v6: continue the winner; bridge the heading gap.
#
# Full-run results (366 iters each, canonical spawns unless noted):
#   abort_mouth (v4)  obey 0.77/0.78  succ 0.62/0.73   <- best; transfer to the
#                     canonical start began only after ~iteration 150 (it was
#                     0.00 through 140), so the early "plateaued, no transfer"
#                     read was simply premature. Trend at cutoff: still edging up.
#   fixed_lowpen (v3) obey 0.83/0.93  succ 0.60/0.59   all-canonical; recovered
#                     late (succ 0.29/0.45 -> 0.60/0.59 over the last 90 iters),
#                     also still rising at cutoff, but carries the per-step-tax
#                     pathology that twice destroyed traversal elsewhere.
#   curric (v5)       obey 0.00/1.00  succ 0.00/0.93   the reverse curriculum
#                     BACKFIRED -- and localized the residual failure exactly.
#                     Late in training its curriculum spawns sit at the canonical
#                     POSITION but at the path-tangent 45 deg heading (the
#                     curriculum walked position back, never heading), and those
#                     episodes obey ~1.0 while identical 0-deg spawns obey 0.00.
#                     The whole remaining gap is the initial TURN.
#
# So v6:
#   cont      abort_mouth recipe, warm-started from its own checkpoint --
#             load_trunk copies same-shape tensors wholesale, so the learned
#             route weights carry over and this is a straight continuation with
#             fresh data order. Tests whether the climb resumes.
#   cont_yaw  same, plus uniform +/-0.5 rad heading jitter on navigator spawns.
#             Bridges the 45-vs-0 deg gap by sampling the whole fan, and a real
#             RoboMaster is never placed at exactly 0 deg anyway, so this is
#             deployment robustness, not a crutch. Canonical metrics include the
#             jitter, i.e. they get strictly harder to game.
#
#   bash spike/launch_route_obey.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
mkdir -p runs/route_obey_v6

TRUNK=runs/route_obey_v4/abort_mouth.pt

launch () {  # name gpu jitter
  CUDA_VISIBLE_DEVICES="$2" nohup python -u rl/train_route_obey.py \
    --init_nav "$TRUNK" \
    --route_abort_wrong 1 \
    --rew_wrong_corridor 0.0 \
    --route_shaping 1 \
    --spawn_mouths 0.5 \
    --spawn_yaw_jitter "$3" \
    --explore_window 0 \
    --seed 2 \
    --log_csv "runs/route_obey_v6/$1.csv" \
    --run_json "runs/route_obey_v6/$1.json" \
    --save "runs/route_obey_v6/$1.pt" \
    > "runs/route_obey_v6/$1.log" 2>&1 &
  echo "launched $1 on GPU $2 (yaw jitter $3 rad) pid $!"
}

if [ "${1:-}" = "cont" ]; then launch cont 1 0.0; fi
if [ "${1:-}" = "cont_yaw" ]; then launch cont_yaw 3 0.5; fi
if [ -z "${1:-}" ]; then launch cont 1 0.0; launch cont_yaw 3 0.5; fi

# v6 outcome: cont PASSED the stage-2 gate (canonical obey 0.96/0.985, succ
# 0.935/0.895, stable over the last 100 iters). cont_yaw plateaued at 0.46
# obey-top -- it warm-started from the weaker abort_mouth checkpoint. The
# robustness track restarts from the PASSING policy instead:
if [ "${1:-}" = "cont_yaw2" ]; then
  TRUNK=runs/route_obey_v6/cont.pt
  launch cont_yaw2 3 0.5
fi
wait || true
