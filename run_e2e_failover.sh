#!/usr/bin/env bash
set -euo pipefail

# End-to-end failover launcher:
# - exit 42: failover-triggered restart required
# - exit 0: normal completion
# - others : real failure
FAILOVER_EXIT_CODE=42

# Configure run identity (used by train_kd.py: args.save_root/args.note)
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-./results}"
RUN_NOTE="${RUN_NOTE:-e2e_failover}"
RUN_DIR="${BASE_SAVE_ROOT}/${RUN_NOTE}"
RESTART_CONFIG_PATH="${RUN_DIR}/restart_config.json"

# Default GPU set for initial boot (before any restart_config exists).
DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-0,1,2,3}"

# Optional restart limit to avoid infinite loops during debugging.
MAX_RESTARTS="${MAX_RESTARTS:-0}"  # 0 means unlimited
RESTART_COUNT=0

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  ./run_e2e_failover.sh <train_kd.py args...>

Example:
  ./run_e2e_failover.sh \
    --data_name cifar100 \
    --t_name resnet110 \
    --s_name resnet20 \
    --kd_mode logits \
    --s_init <student_ckpt> \
    --t_model <teacher_ckpt>

Environment variables:
  BASE_SAVE_ROOT       Base save directory (default: ./results)
  RUN_NOTE             Run note subdir (default: e2e_failover)
  DEFAULT_VISIBLE_GPUS Initial visible GPUs before restart_config exists (default: 0,1,2,3)
  MAX_RESTARTS         Max failover restarts; 0 for unlimited (default: 0)
USAGE
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tspipe
fi

mkdir -p "${RUN_DIR}"

echo "[E2E] BASE_SAVE_ROOT=${BASE_SAVE_ROOT} RUN_NOTE=${RUN_NOTE} RUN_DIR=${RUN_DIR}"
echo "[E2E] Starting failover loop..."

while true; do
  GPU_ASSIGNMENT=""
  NUM_GPUS=0

  if [[ -f "${RESTART_CONFIG_PATH}" ]]; then
    mapfile -t _gpu_meta < <(python - "${RESTART_CONFIG_PATH}" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    gpu_assignment = cfg.get('partition', {}).get('gpu_assignment', [])
    if isinstance(gpu_assignment, list) and gpu_assignment:
        gpu_assignment = [int(x) for x in gpu_assignment]
        print(','.join(str(x) for x in gpu_assignment))
        print(len(gpu_assignment))
    else:
        print('')
        print(0)
except Exception:
    print('')
    print(0)
PY
)
    GPU_ASSIGNMENT="${_gpu_meta[0]:-}"
    NUM_GPUS="${_gpu_meta[1]:-0}"
  fi

  if [[ -n "${GPU_ASSIGNMENT}" && "${NUM_GPUS}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENT}"
    echo "[E2E] Restart config detected -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, nproc=${NUM_GPUS}"
  else
    export CUDA_VISIBLE_DEVICES="${DEFAULT_VISIBLE_GPUS}"
    NUM_GPUS=$(python - <<'PY'
import os
v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
items = [x.strip() for x in v.split(',') if x.strip() != '']
print(len(items) if items else 1)
PY
)
    echo "[E2E] Fresh start/default -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, nproc=${NUM_GPUS}"
  fi

  set +e
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    # For single-process dry-runs, run Python directly so exit code 42 is preserved.
    python benchmarks/soft_target/train_kd.py \
      --tspipe-enable \
      --save_root "${BASE_SAVE_ROOT}" \
      --note "${RUN_NOTE}" \
      "$@"
    EXIT_CODE=$?
  else
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
      benchmarks/soft_target/train_kd.py \
      --tspipe-enable \
      --save_root "${BASE_SAVE_ROOT}" \
      --note "${RUN_NOTE}" \
      "$@"
    EXIT_CODE=$?
  fi
  set -e

  echo "[E2E] train_kd.py exited with code ${EXIT_CODE}"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "[E2E] Training completed normally. Exiting launcher."
    break
  fi

  if [[ "${EXIT_CODE}" -eq "${FAILOVER_EXIT_CODE}" ]]; then
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[E2E] Failover restart requested (code ${FAILOVER_EXIT_CODE}), restart_count=${RESTART_COUNT}"

    if [[ "${MAX_RESTARTS}" -gt 0 && "${RESTART_COUNT}" -ge "${MAX_RESTARTS}" ]]; then
      echo "[E2E] Reached MAX_RESTARTS=${MAX_RESTARTS}. Stopping launcher."
      exit 1
    fi

    sleep 1
    continue
  fi

  echo "[E2E] Unexpected failure exit code ${EXIT_CODE}. Stopping launcher."
  exit "${EXIT_CODE}"
done
