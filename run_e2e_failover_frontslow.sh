#!/usr/bin/env bash
set -euo pipefail

# E2E Failover 테스트 (초기 이후 전체 구간 "강한" 장시간 slowdown, 다중 epoch)
# - GPU: 1,2,5,6 사용 (CUDA_VISIBLE_DEVICES 기준)
# - FAILOVER_SLOWDOWN_THRESHOLD_SEC = 10.0 고정
# - slowdown 구간: baseline 이후 거의 전체 학습 (step 50~4000), factor≈3.0x

# 환경 변수 설정
export DEFAULT_VISIBLE_GPUS=1,2,5,6
export E2E_MASTER_IP=127.0.0.1

# 시나리오 기반(KEEP_REPLAN_DEGRADE 등)은 비우고, 직접 slowdown 파라미터를 넘긴다.
unset FAILOVER_INJECT_SCENARIO
export FAILOVER_TEST_FAST_GATES=0
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=5.0

export RUN_NOTE="e2e_failover_frontslow_1"
export SLOWDOWN_GPU=3  # CUDA_VISIBLE_DEVICES=1,2,5,6 기준 index 3 -> 물리 GPU 6

# E2E 런처 호출
bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size=128 \
  --num_class=100 \
  --epochs=10 \
  --max-steps-per-epoch=0 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu=3 \
  --slowdown-factor=3.0 \
  --slowdown-start=30 \
  --slowdown-end=4000
