#!/usr/bin/env bash
# WP2 — complete the v8 condition sweep + extend the headline to 5 seeds.
#
# New relative to race_v8/:
#   none      : FIXED floor — real anchor, zero content (v8 zeroed the whole
#               wire; same result, 0.488-0.518, but this is the honest control)
#   position  : scout's normalized xy, padded to matched width. Scout is
#               stationary => constant message => must sit at the floor.
#   kinematic : position + constant-velocity extrapolation. Also constant.
#   z_hat     : predicted next latent P(z_t, STAY) — Tier 1's C2.
#   raw_obs   : full 64x64x3 frame + anchor (12290 wire) — share-everything
#               ceiling, deliberately unmatched (Tier 1 design).
#   oracle/z_t seeds 4-5 : headline to 5 seeds.
#
# GPU 0 refuses CUDA init on this box (WP1, 2026-08-05) — streams live on 1-3,
# two per GPU. Each stream is sequential; ~4 h/run, longest stream 4 runs.
#
#   bash spike/launch_race_v8b.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/race_v8b

run () {  # condition seed
  local tag="$1_s$2"
  python -u rl/train_race_route.py \
    --condition "$1" \
    --executor runs/route_obey_v6/cont.pt \
    --total_episodes 6000 \
    --seed "$2" \
    --log_csv "runs/race_v8b/$tag.csv" \
    --run_json "runs/race_v8b/$tag.json" \
    --save "runs/race_v8b/$tag.pt" \
    >> "runs/race_v8b/$tag.log" 2>&1
}

stream () {  # gpu name jobs...
  local gpu="$1" name="$2"; shift 2
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for job in "$@"; do
      run ${job/:/ }   # "cond:seed" -> "cond seed"
      echo "[stream $name] finished $job"
    done
    echo "[stream $name] DONE"
  ) >> "runs/race_v8b/stream_$name.log" 2>&1 &
  echo "stream $name on GPU $gpu pid $! : $*"
}

# gate-critical jobs (none fix, position, kinematic) lead every stream
stream 1 g1a none:1 kinematic:1 raw_obs:1 none:4
stream 1 g1b position:1 z_hat:1 oracle:4
stream 2 g2a none:2 kinematic:2 raw_obs:2 none:5
stream 2 g2b position:2 z_hat:2 z_t:4
stream 3 g3a none:3 kinematic:3 raw_obs:3
stream 3 g3b position:3 z_hat:3 oracle:5 z_t:5
