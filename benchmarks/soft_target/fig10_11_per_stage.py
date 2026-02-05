import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# CONFIGURATION
# ======================
# 실제 데이터가 있는 경로로 수정해주세요
BASE_DIR = Path("results/st")
BATCH_SIZES = [8, 16, 32, 64, 128] 
CSV_NAME = "tspipe_profile.csv"

# ======================
# DATA LOADING
# ======================
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
        
    return pd.read_csv(
        path,
        header=None,
        names=["ts", "batch", "view", "ubatch", "is_target", "src", "dst", "op"]
    )

def normalize_is_target(x):
    if isinstance(x, str):
        return x.upper() == "TRUE"
    return bool(x)

# ======================
# PER-STAGE TIME (핵심 수정됨)
# ======================
def per_stage_time(df):
    """
    배치 하나 내에서 각 Stage(GPU)가 수행한 '유효 작업 시간' 계산
    """
    start = df[~df.op.str.endswith("_finish")].copy()
    end = df[df.op.str.endswith("_finish")].copy()

    start["op_base"] = start.op
    end["op_base"] = end.op.str.replace("_finish", "", regex=False)

    merged = start.merge(
        end,
        on=["batch", "view", "ubatch", "dst", "op_base"],
        suffixes=("_s", "_e")
    )

    # ---------------------------------------------------------
    # [수정 1] 제외 키워드에 'copy' 추가 (Stage 1 H2D Copy 제외)
    # 논문 Eq(16): i != 0 일 때만 RecvActTime 포함
    # wait, barrier: 순수 대기 시간이므로 제외
    # ---------------------------------------------------------
    exclude_keywords = ["wait", "barrier", "copy"]
    
    pattern = "|".join(exclude_keywords)
    # 대소문자 구분 없이 해당 키워드가 포함되면 제외
    mask = ~merged["op_base"].str.contains(pattern, case=False, regex=True)
    
    filtered_df = merged[mask]

    # [수정 2] 시간 단위: 초(Seconds) (ms 변환 제거)
    filtered_df["dur_s"] = (filtered_df.ts_e - filtered_df.ts_s)
    
    return filtered_df.groupby("dst")["dur_s"].sum()

# ======================
# AGGREGATION (Warm-up 추가)
# ======================
def collect_per_stage(csv_path, is_teacher: bool):
    df = load_csv(csv_path)
    if df.empty:
        return None

    df["is_target"] = df["is_target"].apply(normalize_is_target)
    target_df = df[df.is_target == is_teacher]
    
    stage_acc = []

    # [수정 3] Warm-up 적용: 앞쪽 5개 배치는 통계에서 제외
    # (Cold start로 인한 스파이크 제거)
    WARMUP_STEPS = 5
    all_batches = sorted(target_df.batch.unique())
    valid_batches = [b for b in all_batches if b >= WARMUP_STEPS]
    
    # 데이터가 너무 적으면 전체 사용
    if not valid_batches:
        valid_batches = all_batches

    for b in valid_batches:
        batch_df = target_df[target_df.batch == b]
        stage_times = per_stage_time(batch_df)
        
        if len(stage_times) > 0:
            stage_acc.append(stage_times)

    if not stage_acc:
        return None

    return pd.concat(stage_acc, axis=1).mean(axis=1)

# ======================
# PLOTTING
# ======================
def plot_fig(model_name, is_teacher):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    modes = ["baseline", "dpipe"]
    titles = ["(a) Baseline", "(b) Planner"]

    for row, mode in enumerate(modes):
        ax = axes[row]
        data = {}

        for bsz in BATCH_SIZES:
            # 경로 탐색
            path_candidates = [
                BASE_DIR / f"{mode}-profile-b{bsz}" / f"{mode}-profile-b{bsz}" / CSV_NAME,
                BASE_DIR / f"{mode}-profile-b{bsz}" / CSV_NAME
            ]
            
            csv_path = None
            for p in path_candidates:
                if p.exists():
                    csv_path = p
                    break
            
            if csv_path:
                stage_times = collect_per_stage(csv_path, is_teacher)
                if stage_times is not None:
                    data[bsz] = stage_times
            else:
                # 데이터가 없으면 경고 출력
                print(f"[Skip] {mode} B{bsz}: File not found")

        if not data:
            continue

        df_plot = pd.DataFrame(data).T.sort_index()
        df_plot = df_plot.fillna(0)

        # 막대 그래프 그리기
        x = np.arange(len(df_plot.index))
        width = 0.8 / len(df_plot.columns)
        
        for i, stage_id in enumerate(sorted(df_plot.columns)):
            ax.bar(
                x + i * width - 0.4 + width/2, 
                df_plot[stage_id], 
                width=width, 
                label=f"Stage {int(stage_id) + 1}",
                edgecolor='black',
                linewidth=0.5,
                alpha=0.9
            )

        # Y축 라벨 (초 단위)
        ax.set_ylabel("Per-Stage Time (s)")
        ax.set_title(titles[row], fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if row == 0:
            ax.legend(loc='upper right', ncol=min(len(df_plot.columns), 4), fontsize='small')

    if 'df_plot' in locals() and not df_plot.empty:
        axes[1].set_xticks(range(len(df_plot.index)))
        axes[1].set_xticklabels(df_plot.index)
        axes[1].set_xlabel("Batch Size", fontsize=11)

    plt.suptitle(f"Per-stage execution time — {model_name}", fontsize=14, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename = f"figure_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Figure 10: ViT-Large (Teacher)
    print("Plotting Figure 10...")
    plot_fig("ViT-Large (Teacher)", is_teacher=True)

    # Figure 11: ResNet-152 (Student)
    print("Plotting Figure 11...")
    plot_fig("ResNet-152 (Student)", is_teacher=False)