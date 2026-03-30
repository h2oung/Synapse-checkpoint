#!/usr/bin/env bash
set -euo pipefail

# E2E Failover 테스트 (초기 이후 전체 구간 "강한" 장시간 slowdown, 다중 epoch)
# - GPU: 1,2,5,6 사용 (CUDA_VISIBLE_DEVICES 기준)
# - FAILOVER_SLOWDOWN_THRESHOLD_SEC = 10.0 고정
# - slowdown 구간: baseline 이후 거의 전체 학습 (step 50~4000), factor≈3.0x

# 환경 변수 설정
export DEFAULT_VISIBLE_GPUS=0,1,2,4
export E2E_MASTER_IP=127.0.0.1

# 시나리오 기반(KEEP_REPLAN_DEGRADE 등)은 비우고, 직접 slowdown 파라미터를 넘긴다.
unset FAILOVER_INJECT_SCENARIO
export FAILOVER_TEST_FAST_GATES=0
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=5.0

export RUN_NOTE="e2e_failover_0330_2"

# 현재 train_kd.py 구현은 inject-slowdown-gpu를 실제 GPU index가 아니라
# tspipe_trainer.rank와 비교하고 있음.
# 현재 실행 로그상 rank=0 이므로, 실제 sleep 주입을 확인하려면 0으로 맞춰야 함.
export SLOWDOWN_GPU=0

bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size=32 \
  --num_class=100 \
  --epochs=50 \
  --max-steps-per-epoch=0 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-factor=15 \
  --slowdown-start=60 \
  --slowdown-end=10000