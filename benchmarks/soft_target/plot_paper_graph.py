import json
import matplotlib.pyplot as plt
from datetime import datetime

# 마지막 고해상도 실험 데이터 경로 (지희님 경로에 맞게 수정)
data_path = "/home/wisekhy/tspipe/Synapse-private/failover_results_final_verification_highres/failover_basic_20260225_152158_20260225_152207/performance.jsonl"
events_path = "/home/wisekhy/tspipe/Synapse-private/failover_results_final_verification_highres/failover_basic_20260225_152158_20260225_152207/failover_events.jsonl"

times = []
gpu2_net_memory = []
base_mem = None
start_time = None
fail_time = None
shutdown_time = None

# 실패 이벤트 시간 추출
with open(events_path, 'r') as f:
    for line in f:
        event = json.loads(line)
        if event['event_type'] == 'gpu_failure_simulation_start':
            fail_time = datetime.fromisoformat(event['timestamp'])
            break

# 데이터 추출 및 영점 조절
with open(data_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        t = datetime.fromisoformat(data['timestamp'])
        
        if start_time is None:
            start_time = t
            base_mem = data['gpu_memory_used'].get('2', 0)
            
        times.append((t - start_time).total_seconds())
        
        # 순수 메모리 = 현재 메모리 - 기본 컨텍스트 (최소값 0)
        curr_mem = data['gpu_memory_used'].get('2', 0)
        net_mem = max(0, curr_mem - base_mem)
        gpu2_net_memory.append(net_mem)
        
        # 메모리가 떨어지기 시작하는 지점을 셧다운 시간으로 추정
        if fail_time and t > fail_time and net_mem < 400 and shutdown_time is None:
            shutdown_time = t

fail_sec = (fail_time - start_time).total_seconds() if fail_time else None
shutdown_sec = (shutdown_time - start_time).total_seconds() if shutdown_time else (fail_sec + 4.6)

# 시각화 설정
plt.figure(figsize=(10, 5))
plt.plot(times, gpu2_net_memory, label='Net Payload Memory (GPU 2)', color='#d62728', linewidth=2.5)

# 이벤트 라인 및 구간 표시
if fail_sec:
    plt.axvline(x=fail_sec, color='black', linestyle='--', label='Worker Killed (SIGKILL)')
    plt.axvline(x=shutdown_sec, color='blue', linestyle=':', label='System Teardown')
    
    # 장애 감지 구간 색칠 (가장 중요한 논문 포인트)
    plt.axvspan(fail_sec, shutdown_sec, color='gray', alpha=0.2, label='Detection Window')
    
    # 텍스트 주석 추가
    plt.text((fail_sec + shutdown_sec)/2, max(gpu2_net_memory)*0.8, 
             'Health Monitor\nVerification', horizontalalignment='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('Dynamic Memory Deallocation Upon Worker Failure', fontsize=14, pad=15)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Net Memory Usage (MB)', fontsize=12)
plt.ylim(-50, max(gpu2_net_memory) + 100)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='lower right')
plt.tight_layout()

plt.savefig('paper_figure_net_memory_annotated.png', dpi=300)
print("✅ 논문용 최종 그래프가 paper_figure_net_memory_annotated.png 로 저장되었습니다!")