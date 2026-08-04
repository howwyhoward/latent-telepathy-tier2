#!/usr/bin/env bash
# Race v7: identical to v6 except the navigator's exploration now reaches the
# other corridor. v6 was a clean null (oracle 0.44/0.53 vs none 0.50) because
# the top branch was sampled 0.00 of the time from the canonical start
# (spike/diag_exploration.py); an unsampled action cannot acquire a message.
# Schedule: first 40 steps of each episode get correlated (tau=30) lateral
# noise at std 4.48, measured at 0.22 top-corridor coverage for 0.51 success
# vs 0.55 unperturbed, decaying to the trained std over 60% of training so the
# final numbers are on-policy.
#
#   bash spike/launch_race_v7.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
mkdir -p runs/race_v7

TRUNK=runs/nav_pretrain/nav_s1_mouth.pt
GPUS=(1 2 3)
CONDS=(oracle none z_t)

for i in "${!CONDS[@]}"; do
  c="${CONDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" nohup python -u rl/train_race.py \
    --condition "$c" \
    --init_nav "$TRUNK" \
    --spawn_curriculum 0.0 \
    --explore_window 40 \
    --explore_log_std 1.5 \
    --explore_tau 30 \
    --explore_dims y \
    --explore_anneal_frac 0.6 \
    --seed 1 \
    --log_csv "runs/race_v7/${c}.csv" \
    --run_json "runs/race_v7/${c}.json" \
    --save "runs/race_v7/${c}.pt" \
    > "runs/race_v7/${c}.log" 2>&1 &
  echo "launched $c on GPU ${GPUS[$i]} (pid $!)"
done
# One condition dying (e.g. another tenant filling its GPU) must not take the
# launcher, and with it the surviving runs' job control, down with it.
wait || true
