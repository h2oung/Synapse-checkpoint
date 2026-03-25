#!/usr/bin/env bash
set -euo pipefail

# Moderate Persistent Slowdown (REPLAN) 실험 전용 런처
# 기존 run_e2e_failover.sh 구조 기반, 독립적 실행 가능

# 환경 변수 (필요시 수정)
export DEFAULT_VISIBLE_GPUS=0,2,3,5
export E2E_MASTER_IP=127.0.0.1
export FAILOVER_INJECT_SCENARIO=REPLAN_SLOWDOWN
export FAILOVER_TEST_FAST_GATES=1

# 실험 파라미터
BASE_SAVE_ROOT="./results"
RUN_NOTE="e2e_failover_REPLAN"
RUN_DIR="${BASE_SAVE_ROOT}/${RUN_NOTE}"
SLOWDOWN_GPU=2           # slowdown 적용할 GPU 번호
SLOWDOWN_FACTOR=1.5      # slowdown 배수 (1.4~1.6)
SLOWDOWN_DURATION=9999   # slowdown 지속 step 수 (학습 끝까지)

mkdir -p "${RUN_DIR}"


# 재시작 config 경로 (run_e2e_failover.sh와 동일)
SOFT_RESTART_CONFIG_PATH="${RUN_DIR}/failover_restart_config.json"
HARD_RESTART_CONFIG_PATH="${RUN_DIR}/emergency_restart_config.json"
LEGACY_RESTART_CONFIG_PATH="${RUN_DIR}/restart_config.json"

# E2E 실험에서 sustained slowdown threshold를 1초로 강제 (30초 제한 방지)
if [[ -z "${FAILOVER_SLOWDOWN_THRESHOLD_SEC:-}" && -n "${FAILOVER_INJECT_SCENARIO:-}" ]]; then
    export FAILOVER_SLOWDOWN_THRESHOLD_SEC="1.0"
    echo "[E2E] Using fast slowdown threshold: 1.0s (for quick testing with scenario injection)"
fi

RESTART_COUNT=0
FAILOVER_EXIT_CODE=42

while true; do
  # NCCL 포트 관리 (재시작 시 충돌 방지)
  _port_base=$((31200 + (RESTART_COUNT * 100)))
  export PYTORCH_DISTRIBUTED_NCCL_START_PORT=${_port_base}

  # 장애/재시작 config 적용
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

  if [[ -n "${GPU_ASSIGNMENT:-}" && "${NUM_GPUS:-0}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENT}"
    echo "[REPLAN] Restart config detected (${RESTART_SOURCE_PATH}) -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  else
    export CUDA_VISIBLE_DEVICES="${DEFAULT_VISIBLE_GPUS}"
    NUM_GPUS=$(python - <<'PY'
import os
v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
items = [x.strip() for x in v.split(',') if x.strip()]
print(len(items) if items else 1)
PY
    )
    echo "[REPLAN] Fresh start/default -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, partitions=${NUM_GPUS}"
  fi

  # 실험 실행
  set +e
  python benchmarks/soft_target/train_kd.py \
    --tspipe-enable \
    --tspipe-config=benchmarks/soft_target/tspipe.yaml \
    --ip="${E2E_MASTER_IP}" \
    --rank=0 \
    --num-nodes=1 \
    --save_root "${BASE_SAVE_ROOT}" \
    --note "${RUN_NOTE}" \
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
    --max-steps-per-epoch=200 \
    --inject-slowdown-gpu=${SLOWDOWN_GPU} \
    --slowdown-factor=${SLOWDOWN_FACTOR} \
    --slowdown-duration=${SLOWDOWN_DURATION}
  EXIT_CODE=$?
  set -e

  echo "[REPLAN] train_kd.py exited with code ${EXIT_CODE}"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "[REPLAN] Training completed normally. Exiting launcher."
    break
  fi

  if [[ "${EXIT_CODE}" -eq "${FAILOVER_EXIT_CODE}" ]]; then
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[REPLAN] Failover restart requested (code ${FAILOVER_EXIT_CODE}), restart_count=${RESTART_COUNT}"
    sleep 1
    continue
  fi

  echo "[REPLAN] Unexpected failure exit code ${EXIT_CODE}. Stopping launcher."
  exit "${EXIT_CODE}"
done
