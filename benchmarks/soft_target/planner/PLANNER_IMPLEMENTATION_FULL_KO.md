# Planner 수학적 Failover 정책 통합 구현 문서 (최신 동기화)

적용 범위:
- `benchmarks/soft_target/planner/*`
- `benchmarks/soft_target/train_kd.py`
- `run_e2e_failover.sh`
- `test_minimal_failover.py`
- `benchmarks/soft_target/planner/logs/eta_breakdown.jsonl`

---

## 1. 현재 시스템 한 줄 요약

현재 구현은 다음 1-cycle을 자동으로 수행한다.

`장애/슬로우다운 감지 -> ETA 기반 정책 선택 -> 실시간 DP 재분할 -> restart_config + 체크포인트 저장 -> Exit Code 42 종료 -> 외부 런처 재시작 -> 부트스트랩 로더 복구 -> 중단 다음 step부터 재개`

---

## 2. 정책 모델과 실험 전제

핵심 정책 선택식:

`ETA(p) = C_restart(p) + K_rem * T(p)`

- `p ∈ {KEEP, REPLAN, DEGRADE}`
- `K_rem`: 남은 step
- `T(p)`: 정책 적용 후 병목 step time
- `C_restart(p)`: 재시작/복구 오버헤드

0-B 규칙(코드 반영):
- `C_load = 4.37s`
- rollback = `50 steps`
- REPLAN/DEGRADE restart overhead = `4.37 + 50 * T_base`

---

## 3. 현재 코드 아키텍처

### 3.1 정책 결정

`benchmarks/soft_target/planner/dynamic_policy_selector.py`

- 지속시간 게이트 + 임계치 + 진행률을 반영
- `stage_time_predictor`와 `eta_calculator`를 묶어 추천 정책 출력

### 3.2 Stage time/분할 계산

`benchmarks/soft_target/planner/stage_time_predictor.py`

- KEEP: 현재 partition 기반 계산
- REPLAN/DEGRADE: DP minimax contiguous partition 기반 계산
- 실행용 API:
  - `solve_optimal_partition(...)`
  - `calculate_partition_bottleneck_time(...)`
  - `_solve_minimax_dp_partition(...)`

### 3.3 실행 엔진

`benchmarks/soft_target/planner/mathematical_optimizer.py`

- 최신 `alpha_comp`, `beta_comm`로 REPLAN/DEGRADE 시점에 DP 즉시 재실행
- 도출 partition을 `current_partition`에 즉시 반영
- ETA breakdown JSONL 기록
- restart 자동화 훅 내장

### 3.4 학습 루프 연동

`benchmarks/soft_target/train_kd.py`

- `RuntimeTimingIngestor`로 profiler JSONL 증분 ingest
- step마다 progress/alpha-beta 갱신
- slowdown 발생 시 정책 결정 + 정책 실행

---

## 4. 완료된 핵심 구현 (기능 기준)

### 4.1 REPLAN/DEGRADE 실시간 DP 적용

- 예측 DP와 실제 적용 DP가 분리되지 않도록 실행 시점 재최적화로 통일
- 기존 equal repartition 경로는 실행 경로에서 제거

### 4.2 재시작 기반 자동화(Method 1)

- REPLAN/DEGRADE 확정 직후 자동으로 다음 수행:
  1. `restart_config.json` overwrite 저장
  2. 강제 failover 체크포인트 저장
  3. 프로세스 종료

- 종료 코드는 failover 전용 `SystemExit(42)` 사용

### 4.3 부트스트랩 로더 복구

학습 시작 시 `args.save_root/restart_config.json` 검사:

1. 존재 시 `--tspipe-config` YAML의 `tspipe.model_split`을 restart partition으로 덮어씀
2. `checkpoint_path` 또는 `failover_checkpoint_latest.pth` 로드
3. 모델/옵티마이저 state 복원
4. `global_step` 기반 `start_epoch`, `resume_batch_offset` 재구성
5. 중단 다음 배치부터 재개

복구 성공 로그:

`Failover recovery successful. Resuming training from step [X] with new partition.`

### 4.4 외부 런처 자동 재기동

`run_e2e_failover.sh`

- `while true` 루프 기반 런처
- `restart_config.json`의 `gpu_assignment` 길이로 `NUM_GPUS` 동적 계산
- `CUDA_VISIBLE_DEVICES` 동적 동기화
- 종료 코드 처리:
  - `42`: failover 재시작
  - `0`: 정상 완료 종료
  - 그 외: 오류 종료

---

## 5. 파일별 최종 상태

### 5.1 `benchmarks/soft_target/planner/mathematical_optimizer.py`

핵심:
- Phase-0 baseline freeze
- ratio-only alpha/beta runtime 업데이트
- ETA breakdown JSONL
- 실시간 DP 재분할 실행
- restart 자동화 API/트리거

핵심 함수:
- `configure_failover_restart(...)`
- `_build_restart_payload(...)`
- `_write_restart_config(...)`
- `_trigger_failover_restart(...)`
- `_execute_replan(...)`
- `_execute_degrade(...)`

### 5.2 `benchmarks/soft_target/planner/stage_time_predictor.py`

핵심:
- REPLAN/DEGRADE DP 기반 stage time 계산
- 실제 적용 가능한 partition 자체를 반환하는 API 제공

### 5.3 `benchmarks/soft_target/train_kd.py`

핵심:
- failover restart 콜백 연결
- 부트스트랩 로더 구현
- 중간 epoch 배치 스킵 재개 구현
- E2E 검증용 dry-run 모드 추가 (`--dryrun-failover-cycle`)

### 5.4 `run_e2e_failover.sh`

핵심:
- 재시작 루프
- GPU 수 동기화
- 종료 코드 기반 분기
- `nproc=1`에서 python 직접 실행(Exit 42 보존)

---

## 6. 실행 방법 (실무용)

### 6.1 단위 회귀 테스트

```bash
conda activate tspipe
python test_minimal_failover.py
```

성공 기준:
- `PASS: KEEP -> REPLAN -> DEGRADE scenario verified`

### 6.2 E2E Failover 드라이런 (권장 1-cycle 검증)

```bash
conda activate tspipe
BASE_SAVE_ROOT=./results RUN_NOTE=e2e_failover_dryrun DEFAULT_VISIBLE_GPUS=0 MAX_RESTARTS=3 \
./run_e2e_failover.sh \
  --dryrun-failover-cycle \
  --data_name cifar100 \
  --t_name resnet110 \
  --s_name resnet20 \
  --kd_mode logits \
  --s_init /tmp/dummy_s.pth \
  --t_model /tmp/dummy_t.pth
```

검증 포인트(로그):
1. `Failover triggered (Policy: ...)` + `exited with code 42`
2. `[E2E] Failover restart requested (code 42)`
3. `[E2E] Restart config detected -> CUDA_VISIBLE_DEVICES=..., nproc=...`
4. `Failover recovery successful. Resuming training from step [...]`
5. `Resume progress: step A -> B`
6. `[E2E] Training completed normally. Exiting launcher.`

### 6.3 실학습 E2E 실행

`run_e2e_failover.sh`에 실제 학습 인자/체크포인트 경로를 넣어 실행한다.

```bash
conda activate tspipe
BASE_SAVE_ROOT=./results RUN_NOTE=e2e_failover_real DEFAULT_VISIBLE_GPUS=0,1,2,3 \
./run_e2e_failover.sh \
  --data_name <dataset> \
  --t_name <teacher_model> \
  --s_name <student_model> \
  --kd_mode logits \
  --s_init <student_init_ckpt> \
  --t_model <teacher_ckpt>
```

---

## 7. 출력 산출물

실행 중 생성되는 핵심 파일:

- `args.save_root/restart_config.json`
  - trigger policy
  - partition(`gpu_assignment`, `snet_partition`, `tnet_partition`, boundaries)
  - alpha/beta
  - restart overhead metadata
- `args.save_root/failover_checkpoint_latest.pth`
  - student/teacher/optimizer state
  - `global_step`
- `benchmarks/soft_target/planner/logs/eta_breakdown.jsonl`
  - 정책 근거(ETA/오버헤드/계수) 증적

---

## 8. 현재 제약/주의사항

1. `--tspipe-config`가 없으면 YAML overwrite는 스킵되며 경고 로그가 남는다.
2. 분산 런처 환경별로 자식 종료코드 전파 특성이 다를 수 있다.
3. 장기 실험에서 재현성 완전 일치를 원하면 DataLoader/샘플러 상태 저장까지 추가 권장.
4. 로그 파일은 과거 실행 기록과 혼재될 수 있으므로 최근 타임스탬프 기준 확인 권장.

---

## 9. 팀 온보딩용 빠른 체크리스트

1. `conda activate tspipe`
2. `python test_minimal_failover.py` 통과 확인
3. `run_e2e_failover.sh`로 dry-run 1-cycle 수행
4. 로그에서 42 종료 -> 재시작 -> recovery -> resume 진행 확인
5. 실제 학습 인자로 E2E 실행

---

## 부록 A. 주요 파일

- `benchmarks/soft_target/planner/mathematical_optimizer.py`
- `benchmarks/soft_target/planner/stage_time_predictor.py`
- `benchmarks/soft_target/planner/eta_calculator.py`
- `benchmarks/soft_target/planner/dynamic_policy_selector.py`
- `benchmarks/soft_target/train_kd.py`
- `run_e2e_failover.sh`
- `test_minimal_failover.py`
- `benchmarks/soft_target/planner/logs/eta_breakdown.jsonl`
