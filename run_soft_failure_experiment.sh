#!/bin/bash

set -e

# ============================================================================
# Soft Failure Experiment Script
# Reference: Working baseline before failover was added
# Teacher: ViT Large (model_best.pth.tar)
# Student: ResNet152 (initial_r152.pth.tar)
# Dataset: ImageNet100 (from /nas-ssd/datasets/imagenet2012/imagenet)
# GPUs: 0, 3, 4, 6
# ============================================================================

PROJECT_ROOT="/acpl-ssd10/Synapse-private"
RESULTS_DIR="${PROJECT_ROOT}/results/soft_failure_experiment_$(date +%Y%m%d_%H%M%S)"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmarks/soft_target"

# Model paths (relative to BENCHMARK_DIR)
TEACHER_MODEL="./results/base/base-i100-vit-large/model_best.pth.tar"
STUDENT_MODEL="./results/base/base-i100-resnet152/initial_r152.pth.tar"

# Dataset path
IMG_ROOT="/nas-ssd/datasets/imagenet2012/imagenet"

# Create results directory
mkdir -p "${RESULTS_DIR}"
echo "📁 Results directory: ${RESULTS_DIR}"

# ============================================================================
# Verify model files exist
# ============================================================================
cd "$BENCHMARK_DIR"

if [ ! -f "$TEACHER_MODEL" ]; then
    echo "❌ Teacher model not found: $TEACHER_MODEL"
    exit 1
fi

if [ ! -f "$STUDENT_MODEL" ]; then
    echo "❌ Student model not found: $STUDENT_MODEL"
    exit 1
fi

echo "✅ Teacher model found: $TEACHER_MODEL"
echo "✅ Student model found: $STUDENT_MODEL"

# ============================================================================
# Run training with TSPipe + Soft Failover monitoring
# ============================================================================

# Set GPU devices (0, 3, 4, 6)
export CUDA_VISIBLE_DEVICES=0,3,4,6

echo ""
echo "=========================================="
echo "🚀 Starting Soft Failure Experiment"
echo "=========================================="
echo "Teacher: ViT Large"
echo "Student: ResNet152"
echo "Dataset: ImageNet100"
echo "Mode: TSPipe + Soft Failover"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "==========================================\n"

python train_kd.py \
  --img_root="${IMG_ROOT}" \
  --data_name=imagenet100 \
  --num_class=100 \
  --t_name=vit_large \
  --s_name=resnet152 \
  --t_model="${TEACHER_MODEL}" \
  --s_init="${STUDENT_MODEL}" \
  --kd_mode=st \
  --lambda_kd=0.1 \
  --T=4.0 \
  --epochs=1 \
  --batch_size=128 \
  --save_root="${RESULTS_DIR}/soft_failure_test" \
  --note=soft-failure-tspipe-gpu0346-b128 \
  --tspipe-enable \
  --tspipe-config=./tspipe.yaml \
  --ip=127.0.0.1 \
  --rank=0 \
  --num-nodes=1 \
  2>&1 | tee "${RESULTS_DIR}/run.log"

# ============================================================================
# Verify experiment results
# ============================================================================

echo ""
echo "=========================================="
echo "✅ Soft Failure Experiment Completed"
echo "=========================================="

if [ -f "${RESULTS_DIR}/soft_failure_test/failover_restart_config.json" ]; then
    echo "✅ Soft failover config saved:"
    echo "   ${RESULTS_DIR}/soft_failure_test/failover_restart_config.json"
    cat "${RESULTS_DIR}/soft_failure_test/failover_restart_config.json" | head -20
fi

if [ -f "${RESULTS_DIR}/soft_failure_test/failover_checkpoint_latest.pth" ]; then
    echo ""
    echo "✅ Failover checkpoint saved:"
    ls -lh "${RESULTS_DIR}/soft_failure_test/failover_checkpoint_latest.pth"
fi

📊 Experiment artifacts in: ${RESULTS_DIR}/
Log file: ${RESULTS_DIR}/run.log
=========================================="
    echo "   $RESULTS_DIR/failover_checkpoint_latest.pth"
    ls -lh "${RESULTS_DIR}/failover_checkpoint_latest.pth"
fi

echo ""
echo "📊 Experiment artifacts in: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/" | grep -E "\.json|\.pth|\.log"

echo ""
echo "=========================================="
echo "Log file: ${RESULTS_DIR}/run.log"
echo "=========================================="
