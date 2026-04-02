#!/bin/bash

unset FAILOVER_INJECT_SCENARIO
export FAILOVER_TEST_FAST_GATES=0
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=30.0

export RUN_NOTE="e2e_failover_batch32_0402_updated"

export SLOWDOWN_GPU=0

CUDA_VISIBLE_DEVICES=0,3,4,6 \
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
  --epochs=1 \
  --max-steps-per-epoch=0 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-factor=3.0 \
  --slowdown-start=50 \
  --slowdown-end=120
