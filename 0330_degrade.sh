#!/usr/bin/env bash
set -euo pipefail

# Best-effort DEGRADE probe using the *actual* slowdown injection path.
#
# Important:
# - This does NOT use FAILOVER_INJECT_SCENARIO.
# - In the current main codepath, real slowdown is injected inside gpu_worker.py
#   via --inject-slowdown-gpu / --slowdown-factor / --slowdown-start / --slowdown-end
#   / --slowdown-task-scope.
# - In existing logs, clean-start soft slowdown often chose REPLAN first.
# - A DEGRADE decision became more plausible after repartition/restart state had
#   already changed the active partition and learned alpha/beta.
#
# So this script is tuned to maximize the chance of seeing:
#   KEEP -> REPLAN -> DEGRADE
# or at least a later DEGRADE after one restart,
# rather than guaranteeing DEGRADE on the very first trigger.

export DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-1,3,5,6}"
export E2E_MASTER_IP="${E2E_MASTER_IP:-127.0.0.1}"
unset FAILOVER_INJECT_SCENARIO

# Faster probe gating than the original 30s production-like setting.
# We want the first few decisions quickly while still using the real slowdown path.
export FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:-1}"
export FAILOVER_SLOWDOWN_THRESHOLD_SEC="${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-1.0}"

# Allow a couple of failover restarts because DEGRADE may emerge only after an
# earlier REPLAN changes the partition/coefficient state.
export MAX_RESTARTS="${MAX_RESTARTS:-3}"

export RUN_NOTE="${RUN_NOTE:-e2e_failover_degrade_actual_probe}"

# Use a middle local GPU instead of GPU0.
# Existing logs suggest front/last-stage slowdown tends to favor REPLAN,
# while a middle-stage slowdown is more likely to make K-1 repartition attractive.
export SLOWDOWN_GPU="${SLOWDOWN_GPU:-1}"

# Strong and persistent slowdown over both compute and communication tasks.
export SLOWDOWN_FACTOR="${SLOWDOWN_FACTOR:-3.0}"
export SLOWDOWN_START="${SLOWDOWN_START:-55}"
export SLOWDOWN_END="${SLOWDOWN_END:-4000}"
export SLOWDOWN_TASK_SCOPE="${SLOWDOWN_TASK_SCOPE:-both}"

# Batch 64 is a better probe than batch 16 for DEGRADE in the current logs.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-2}"
export MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size="${BATCH_SIZE}" \
  --num_class=100 \
  --epochs="${EPOCHS}" \
  --max-steps-per-epoch="${MAX_STEPS_PER_EPOCH}" \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-task-scope="${SLOWDOWN_TASK_SCOPE}" \
  --slowdown-factor="${SLOWDOWN_FACTOR}" \
  --slowdown-start="${SLOWDOWN_START}" \
  --slowdown-end="${SLOWDOWN_END}"
