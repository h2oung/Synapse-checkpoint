#!/usr/bin/env bash
set -euo pipefail

# E2E Failover 테스트 (KEEP_REPLAN_DEGRADE 시나리오)
# slowdown factor ~1.1~1.2x, duration ~50 steps 목적

# 환경 변수 설정

export DEFAULT_VISIBLE_GPUS=1,3,5,6
export E2E_MASTER_IP=127.0.0.1
export FAILOVER_INJECT_SCENARIO=KEEP_REPLAN_DEGRADE
export FAILOVER_TEST_FAST_GATES=1
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=10.0
export RUN_NOTE="e2e_failover_keep"
export SLOWDOWN_GPU=0  # 단일 프로세스 환경에서 slowdown이 정상 주입되도록 0으로 설정

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
  --epochs=1 \
  --max-steps-per-epoch=0 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml
