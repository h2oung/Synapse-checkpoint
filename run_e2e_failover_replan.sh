#!/usr/bin/env bash
set -euo pipefail

# E2E Failover 테스트 (REPLAN 시나리오)
# slowdown factor 1.4~1.6x, duration=training 끝까지, REPLAN 정책 유도 목적

# 환경 변수 설정
export DEFAULT_VISIBLE_GPUS=1,2,3,5
export E2E_MASTER_IP=127.0.0.1
export FAILOVER_INJECT_SCENARIO=REPLAN_SLOWDOWN
export FAILOVER_TEST_FAST_GATES=1
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=1.0
export RUN_NOTE="e2e_failover_replan4"

# 실험 파라미터 (slowdown 관련)
export SLOWDOWN_GPU=0           # slowdown 적용할 GPU 번호 (단일 프로세스 실험에서는 0으로 고정)
export SLOWDOWN_FACTOR=8.0     # slowdown 배수 (1.4~1.6)
export SLOWDOWN_DURATION=9999   # slowdown 지속 step 수 (학습 끝까지)

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
    --batch_size=64 \
    --num_class=100 \
    --epochs=1 \
    --max-steps-per-epoch=0 \
    --tspipe-enable \
    --tspipe-config=benchmarks/soft_target/tspipe.yaml \
    --inject-slowdown-gpu=${SLOWDOWN_GPU} \
    --slowdown-factor=${SLOWDOWN_FACTOR} \
    --slowdown-start=50 \
    --slowdown-duration=${SLOWDOWN_DURATION}
