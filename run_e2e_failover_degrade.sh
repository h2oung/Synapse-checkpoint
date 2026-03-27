#!/usr/bin/env bash
set -euo pipefail

# E2E Failover 테스트 (Severe Persistent Slowdown -> DEGRADE 유도)
# 목적:
# - 특정 GPU를 사실상 unusable 수준(3x~5x)으로 느리게 만들어 DEGRADE 선택 유도
# - train_kd.py의 실제 wall-clock slowdown 주입(time.sleep) 경로 사용

# GPU/런타임 기본 설정 (필요 시 외부에서 override 가능)
export DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-1,3,5,6}"
export E2E_MASTER_IP="${E2E_MASTER_IP:-127.0.0.1}"

# 실험 라벨
export RUN_NOTE="${RUN_NOTE:-e2e_failover_degrade}"

# 시나리오 라벨 (로깅용)
export FAILOVER_INJECT_SCENARIO="${FAILOVER_INJECT_SCENARIO:-DEGRADE_SLOWDOWN}"

# 중요: sustained gate를 비활성화해 KEEP 고착을 방지
# (0.0이면 즉시 ETA 기반 정책 평가)
export FAILOVER_SLOWDOWN_THRESHOLD_SEC="${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-0.0}"

# 디버그/테스트 게이트 (필요 시 유지)
export FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:-1}"

# Severe slowdown 설정 (persistent)
export SLOWDOWN_GPU="${SLOWDOWN_GPU:-0}"
export SLOWDOWN_FACTOR="${SLOWDOWN_FACTOR:-4.0}"
export SLOWDOWN_START="${SLOWDOWN_START:-30}"
export SLOWDOWN_END="${SLOWDOWN_END:-99999}"

# 모델 경로 (환경에 따라 override 권장)
export T_MODEL_PATH="${T_MODEL_PATH:-/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar}"
export S_INIT_PATH="${S_INIT_PATH:-/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar}"

bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model="${T_MODEL_PATH}" \
  --s_init="${S_INIT_PATH}" \
  --batch_size=128 \
  --num_class=100 \
  --epochs=1 \
  --max-steps-per-epoch=220 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-factor="${SLOWDOWN_FACTOR}" \
  --slowdown-start="${SLOWDOWN_START}" \
  --slowdown-end="${SLOWDOWN_END}"
