import matplotlib.pyplot as plt
import os

# 데이터
N = [25, 50, 100, 200, 400]
avg_step_time_ms = [424.45, 406.85, 404.00, 536.34, 539.35]
csave_sec = [2.42, 2.41, 2.60, 2.58, 2.53]
pure_overhead_ms_per_step = [96.8, 48.2, 26.0, 12.9, 6.3]
overhead_ratio_percent = [23.08, 11.65, 6.49, 2.43, 0.95]

# 정규화 함수 (min-max normalization)
def min_max_normalize(values):
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.0 for _ in values]  # 값이 전부 같을 때
    return [(v - vmin) / (vmax - vmin) for v in values]

# 정규화된 데이터
avg_step_time_norm = min_max_normalize(avg_step_time_ms)
csave_sec_norm = min_max_normalize(csave_sec)
pure_overhead_norm = min_max_normalize(pure_overhead_ms_per_step)
overhead_ratio_norm = min_max_normalize(overhead_ratio_percent)

# 저장 폴더
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 그래프 생성
plt.figure(figsize=(10, 6))

plt.plot(N, avg_step_time_norm, marker='o', label='Avg Step Time (normalized)')
plt.plot(N, csave_sec_norm, marker='s', label='Actual Save Time (normalized)')
plt.plot(N, pure_overhead_norm, marker='^', label='Pure Save Overhead (normalized)')
plt.plot(N, overhead_ratio_norm, marker='d', label='Overhead Ratio (normalized)')

# 값 표시 (정규화 값)
for x, y in zip(N, avg_step_time_norm):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

for x, y in zip(N, csave_sec_norm):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

for x, y in zip(N, pure_overhead_norm):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

for x, y in zip(N, overhead_ratio_norm):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

plt.xlabel("Checkpoint Interval (N)")
plt.ylabel("Normalized Value (0 ~ 1)")
plt.title("Normalized Trend Comparison of Checkpoint Metrics")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 파일 저장
save_path = os.path.join(output_dir, "normalized_checkpoint_metrics_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"그래프 저장 완료: {save_path}")