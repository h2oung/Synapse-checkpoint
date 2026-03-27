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
SOFT_RESTART_CONFIG_PATH="${RUN_DIR}/failover_restart_config.json"
HARD_RESTART_CONFIG_PATH="${RUN_DIR}/emergency_restart_config.json"
LEGACY_RESTART_CONFIG_PATH="${RUN_DIR}/restart_config.json"
E2E_MASTER_IP="${E2E_MASTER_IP:-127.0.0.1}"

# Default GPU set for initial boot (before any restart config exists).
DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS:-2,3,4,5}"

# Optional restart limit to avoid infinite loops during debugging.
MAX_RESTARTS="${MAX_RESTARTS:-0}"  # 0 means unlimited

# RESTART_COUNT must be initialized outside the loop to persist across restarts
RESTART_COUNT=0

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  ./run_e2e_failover.sh <train_kd.py args...>

Example:
  ./run_e2e_failover.sh \
    --data_name imagenet100 \
    --t_name vit_large \
    --s_name resnet152 \
    --kd_mode st \
    --s_init ./results/base/base-i100-resnet152/initial_r152.pth.tar \
    --t_model ./results/base/base-i100-vit-large/model_best.pth.tar

Environment variables:
  BASE_SAVE_ROOT           Base save directory (default: ./results)
  RUN_NOTE                 Run note subdir (default: e2e_failover)
  DEFAULT_VISIBLE_GPUS     Initial visible GPUs before restart config exists (default: 0,3,4,6)
  MAX_RESTARTS             Max failover restarts; 0 for unlimited (default: 0)
  FAILOVER_INJECT_SCENARIO Inject synthetic slowdown (e.g., KEEP_REPLAN_DEGRADE)
  FAILOVER_TEST_FAST_GATES Enable fast gate detection for quick testing
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

# Measure end-to-end wall-clock time across all restarts
E2E_START_TIME=$(date +%s)

while true; do
  GPU_ASSIGNMENT=""
  NUM_GPUS=0
  RESTART_SOURCE_PATH=""

  # 🔧 Generate unique NCCL port for each restart iteration to avoid socket reuse conflicts
  # Use larger port spacing (100) to ensure TCP TIME_WAIT (60s) doesn't block port reuse
  # RESTART_COUNT=0: 31200 (first run)
  # RESTART_COUNT=1: 31300 (failover restart #1, avoids TIME_WAIT overlap)
  # RESTART_COUNT=2: 31400 (failover restart #2), etc.
  if [[ -n "${E2E_NCCL_BASE_PORT:-}" ]]; then
    _port_base="${E2E_NCCL_BASE_PORT}"
    echo "[E2E] Using preset NCCL base port: ${_port_base}"
  else
    _port_base=$((31200 + (RESTART_COUNT * 100)))
    echo "[E2E] 🔌 Port allocation: RESTART_COUNT=${RESTART_COUNT}, computed port=${_port_base} (spacing=100 to avoid TIME_WAIT)"
    if [[ ${RESTART_COUNT} -gt 0 ]]; then
      echo "[E2E] ✅ Failover restart iteration ${RESTART_COUNT}: Using NCCL port offset for clean socket state"
    fi
  fi

  if [[ -f "${SOFT_RESTART_CONFIG_PATH}" || -f "${HARD_RESTART_CONFIG_PATH}" || -f "${LEGACY_RESTART_CONFIG_PATH}" ]]; then
    mapfile -t _gpu_meta < <(DEFAULT_VISIBLE_GPUS="${DEFAULT_VISIBLE_GPUS}" python - "${SOFT_RESTART_CONFIG_PATH}" "${HARD_RESTART_CONFIG_PATH}" "${LEGACY_RESTART_CONFIG_PATH}" <<'PY'
import json
import os
import sys

def parse_default_visible_gpus():
    raw = os.environ.get("DEFAULT_VISIBLE_GPUS", "")
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except Exception:
            continue
    return vals

def read_gpu_assignment(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        gpu_assignment = cfg.get('partition', {}).get('gpu_assignment', [])
        if isinstance(gpu_assignment, list) and gpu_assignment:
            return [int(x) for x in gpu_assignment]
    except Exception:
        return None
    return None

candidates = []
default_visible = parse_default_visible_gpus()
for p in sys.argv[1:]:
    if os.path.isfile(p):
        gpus = read_gpu_assignment(p)
        if gpus:
            # REPLAN payload often stores local indices (0..N-1) relative to current visibility.
            # Map those local indices back to physical GPU ids from DEFAULT_VISIBLE_GPUS.
            if default_visible and all(0 <= g < len(default_visible) for g in gpus):
                gpus = [default_visible[g] for g in gpus]
            candidates.append((os.path.getmtime(p), gpus, p))

if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, gpus, src = candidates[0]
    print(','.join(str(x) for x in gpus))
    print(len(gpus))
    print(src)
else:
    print('')
    print(0)
    print('')
PY
)
    GPU_ASSIGNMENT="${_gpu_meta[0]:-}"
    NUM_GPUS="${_gpu_meta[1]:-0}"
    RESTART_SOURCE_PATH="${_gpu_meta[2]:-}"
  fi

  if [[ -n "${GPU_ASSIGNMENT}" && "${NUM_GPUS}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENT}"
    echo "[E2E] Restart config detected (${RESTART_SOURCE_PATH}) -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  else
    export CUDA_VISIBLE_DEVICES="${DEFAULT_VISIBLE_GPUS}"
    NUM_GPUS=$(python - <<'PY'
import os
v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
items = [x.strip() for x in v.split(',') if x.strip()]
print(len(items) if items else 1)
PY
)
    echo "[E2E] Fresh start/default -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  fi

  # NCCL 환경 설정 (로컬 단일 노드 디버깅/교착 회피용)
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
  export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"
  echo "[E2E] Dist env: MASTER_IP=${E2E_MASTER_IP}, NCCL_DEBUG=${NCCL_DEBUG}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}, TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME}"

  # IMPORTANT:
  # TSPipe 자체가 단일 primary 프로세스에서 내부 worker/NCCL/RPC를 초기화합니다.
  # torchrun으로 외부 다중 프로세스를 추가하면 rank 충돌과 wait_ready 교착이 발생할 수 있어
  # E2E 런처는 항상 단일 python 프로세스로 train_kd.py를 실행합니다.

  # Note: NCCL port is now allocated per-restart inside the loop (see below)
  
  # Export failover scenario injection if specified (for SlowdownDetector intra-batch slowdown injection)
  # Note: Must re-export here because subprocess env vars don't propagate to script context
  export FAILOVER_INJECT_SCENARIO="${FAILOVER_INJECT_SCENARIO:=}"
  export FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:=}"
  
  # For E2E quick tests: use shorter slowdown threshold (1 second instead of 30)
  # Set FAILOVER_SLOWDOWN_THRESHOLD_SEC=30 to restore production behavior
  if [[ -z "${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-}" && -n "${FAILOVER_INJECT_SCENARIO}" ]]; then
    export FAILOVER_SLOWDOWN_THRESHOLD_SEC="1.0"
    echo "[E2E] Using fast slowdown threshold: 1.0s (for quick testing with scenario injection)"
  fi
  
  if [[ -n "${FAILOVER_INJECT_SCENARIO}" ]]; then
    echo "[E2E] Failover scenario injection enabled: ${FAILOVER_INJECT_SCENARIO}"
  fi
  
  set +e
  echo "[DEBUG] NCCL PORT: ${_port_base}, RESTART_COUNT: ${RESTART_COUNT}"  # Debug log
  PYTORCH_DISTRIBUTED_NCCL_START_PORT=${_port_base} \
  FAILOVER_INJECT_SCENARIO="${FAILOVER_INJECT_SCENARIO}" \
  FAILOVER_SLOWDOWN_THRESHOLD_SEC="${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-}" \
  FAILOVER_TEST_FAST_GATES="${FAILOVER_TEST_FAST_GATES:-}" \
  python benchmarks/soft_target/train_kd.py \
    --tspipe-enable \
    --tspipe-config=benchmarks/soft_target/tspipe.yaml \
    --ip="${E2E_MASTER_IP}" \
    --rank=0 \
    --num-nodes=1 \
    --save_root "${BASE_SAVE_ROOT}" \
    --note "${RUN_NOTE}" \
    "$@"
  EXIT_CODE=$?
  set -e

  echo "[E2E] train_kd.py exited with code ${EXIT_CODE}"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    E2E_END_TIME=$(date +%s)
    E2E_ELAPSED_SEC=$((E2E_END_TIME - E2E_START_TIME))
    if [[ ${E2E_ELAPSED_SEC} -lt 0 ]]; then
      E2E_ELAPSED_SEC=0
    fi
    E2E_ELAPSED_MIN=$((E2E_ELAPSED_SEC / 60))
    E2E_ELAPSED_REM=$((E2E_ELAPSED_SEC % 60))
    echo "[E2E] Total wall-clock time: ${E2E_ELAPSED_SEC}s (${E2E_ELAPSED_MIN}m ${E2E_ELAPSED_REM}s)"
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
