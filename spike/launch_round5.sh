#!/usr/bin/env bash
# Round 5: attack the objective, not the compute (19 Aug audit).
#
# Evidence driving this design:
#   - 0 of ~5000 logged points across all 14 realcam20 runs beat the shipped
#     r2A_rescue6M (0.840 obey / 0.504 greedy success) on both axes: the
#     search is saturated under the old recipe, but ~half of episodes still
#     fail with a PERFECT route bit, so the task is not.
#   - Obedience was bought with success: r2A success 0.243 -> 0.027 -> 0.260
#     while obedience climbed monotonically. The global wrong-corridor
#     penalty/abort suppress goal-reaching gradients everywhere.
#   - Runs ended stalled, not converged: success +0.13/M over the final 30 %,
#     rolling over as approx_kl collapsed 0.050 -> 0.005 (LR anneal squeezed
#     the trust region to nothing mid-recovery).
#   - Every run peaked mid-run; end-of-run selection threw the peaks away.
#
# Changes, one per lever:
#   ALL:  peak snapshotting is now built into the trainer (*.pt.best is the
#         artifact to eval, never the end-of-run file), LR anneal OFF, 12M
#         steps so the recovery phase can finish.
#   A:    r2A recipe otherwise unchanged -- isolates selection + schedule.
#   B:    success bonus gated on obedience (rew_success_obedient_only), the
#         global wrong-corridor penalty and abort both OFF -- obedient arrival
#         is the only rewarded outcome, and its gradient is never suppressed.
#
# SMOKE FIRST (new env flag has never run):
#   CUDA_VISIBLE_DEVICES=0 python -u rl/train_route_obey.py \
#     --rew_success_obedient_only 1 --route_abort_wrong 0 \
#     --rew_wrong_corridor 0.0 --total_timesteps 40000 --save /tmp/r5smoke.pt
# then confirm it logs iterations and writes /tmp/r5smoke.pt.best.
#
#   bash spike/launch_round5.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source setup/env.sh >/dev/null 2>&1
JEPA=checkpoints/jepa_realcam20.pt          # swap to jepa_realcam20_dr.pt once it wins gate A+C
OUT=runs/realcam20
INIT="$OUT/r2A_rescue6M.pt"

run () {  # name gpu extra...
  local name="$1" gpu="$2"; shift 2
  if [ -f "$OUT/$name.pt" ]; then echo "=== $name: exists, skipping ==="; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u rl/train_route_obey.py \
    --init_nav "$INIT" \
    --jepa_ckpt "$JEPA" \
    --total_timesteps 12000000 \
    --anneal_lr 0 --ent_coef 0.01 \
    --spawn_mouths 0.5 --spawn_yaw_jitter 0.0 --explore_window 0 \
    --seed 1 \
    "$@" \
    --save "$OUT/$name.pt" --log_csv "$OUT/$name.csv" \
    --run_json "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1
  echo "=== $name exit $? ==="
}

# A: old objective, fixed schedule + selection
run r5A_schedule 0 \
  --route_abort_wrong 1 --rew_wrong_corridor 0.0 --route_shaping 1 &

# B: obedient-only success, no penalty, no abort
run r5B_obedient_success 1 \
  --rew_success_obedient_only 1 --route_abort_wrong 0 \
  --rew_wrong_corridor 0.0 --route_shaping 1 &

wait
echo "round 5 done. Greedy-eval the *.pt.best files, not the end-of-run ones:"
echo "  python spike/eval_pixels_to_route.py --policy $OUT/r5A_schedule.pt.best --jepa_ckpt $JEPA"
echo "  python spike/eval_pixels_to_route.py --policy $OUT/r5B_obedient_success.pt.best --jepa_ckpt $JEPA"
