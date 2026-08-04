#!/usr/bin/env bash
# Race v8: recruit the route from the message, over the FROZEN stage-1.5
# executor (runs/route_obey_v6/cont.pt, gate: canonical obey 0.96/0.985).
#
# The only learner is a ~4.5k-parameter route head, one categorical decision
# per episode, trained on episode return alone — a contextual bandit. This is
# the exploration structure v1-v7 lacked: the alternative corridor costs one
# sampled bit, not a 14-sigma Gaussian excursion.
#
# Pre-registered readout (canonical spawns only):
#   oracle : optimization ceiling — must pass or the machinery is broken
#   z_t    : THE thesis condition — reward-only recruitment of the JEPA latent
#   none   : floor — constant preference, route_opt pins ~0.5, hazard ~ half
#            the episodes pay the crossing
#
#   bash spike/launch_race_v8.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
mkdir -p runs/race_v8

SEED="${1:-1}"
SUFFIX=""
[ "$SEED" != "1" ] && SUFFIX="_s$SEED"

launch () {  # condition gpu
  CUDA_VISIBLE_DEVICES="$2" nohup python -u rl/train_race_route.py \
    --condition "$1" \
    --executor runs/route_obey_v6/cont.pt \
    --total_episodes 6000 \
    --seed "$SEED" \
    --log_csv "runs/race_v8/$1$SUFFIX.csv" \
    --run_json "runs/race_v8/$1$SUFFIX.json" \
    --save "runs/race_v8/$1$SUFFIX.pt" \
    > "runs/race_v8/$1$SUFFIX.log" 2>&1 &
  echo "launched race-v8 $1 seed $SEED on GPU $2 pid $!"
}

launch oracle 1
launch z_t 2
launch none 3
wait || true

# Seed-1 outcome (6000 eps each): oracle 0.996 / z_t 0.986 / none 0.518
# route-optimality; z_t hazard 0.00. Reward-only recruitment of the frozen
# JEPA latent, within 1% of the ground-truth ceiling. Seeds 2-3 = replication.
