export E2E_MASTER_IP=127.0.0.1
unset FAILOVER_INJECT_SCENARIO

export FAILOVER_TEST_FAST_GATES=0
export FAILOVER_SLOWDOWN_THRESHOLD_SEC=30
export RUN_NOTE="e2e_failover_degrade_probe"

# 0-based local GPU index
export SLOWDOWN_GPU=3

bash ./run_e2e_failover.sh \
  --img_root=/nas-ssd/datasets/imagenet2012/imagenet \
  --data_name=imagenet100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --t_model=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-vit-large/model_best.pth.tar \
  --s_init=/acpl-ssd10/Synapse-private/benchmarks/results/base/base-i100-resnet152/initial_r152.pth.tar \
  --batch_size=16 \
  --num_class=100 \
  --epochs=2 \
  --max-steps-per-epoch=0 \
  --tspipe-enable \
  --tspipe-config=benchmarks/soft_target/tspipe.yaml \
  --inject-slowdown-gpu="${SLOWDOWN_GPU}" \
  --slowdown-task-scope=both \
  --slowdown-factor=3 \
  --slowdown-start=55 \
  --slowdown-end=10000
