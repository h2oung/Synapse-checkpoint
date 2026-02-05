import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path("results/st")
BATCH_SIZES = [8, 16, 32, 64, 128]
CSV_NAME = "tspipe_profile.csv"
TARGET_ITER = 10


def load_csv(path):
    return pd.read_csv(
        path,
        header=None,
        names=["ts", "batch", "view", "ubatch", "is_target", "src", "dst", "op"],
    )


def classify_role(row):
    op = row["op"].lower()
    is_target = str(row["is_target"]).lower() == "true"

    if "backward" in op:
        return "student"

    if "compute" in op:
        if is_target:
            return "student"
        else:
            return "teacher"

    return None


def stage_time_sum(df, role):
    if TARGET_ITER in df.batch.unique():
        df = df[df.batch == TARGET_ITER]
    else:
        df = df[df.batch == sorted(df.batch.unique())[0]]

    df = df.copy()
    df["role"] = df.apply(classify_role, axis=1)
    df = df[df["role"] == role]

    if df.empty:
        return pd.Series(dtype=float)

    start = df.loc[~df.op.str.endswith("_finish")].copy()
    end = df.loc[df.op.str.endswith("_finish")].copy()

    start["op_base"] = start["op"]
    end["op_base"] = end["op"].str.replace("_finish", "", regex=False)

    merged = start.merge(
        end,
        on=["batch", "view", "ubatch", "dst", "op_base"],
        suffixes=("_s", "_e"),
    )

    if merged.empty:
        return pd.Series(dtype=float)

    merged["dur"] = merged["ts_e"] - merged["ts_s"]

    return merged.groupby("dst")["dur"].sum()


def plot(results, title, save_name):
    stages = results[BATCH_SIZES[0]].index
    x = np.arange(len(BATCH_SIZES))
    w = 0.8 / len(stages)

    plt.figure(figsize=(8, 4))
    for i, s in enumerate(stages):
        plt.bar(
            x + i * w,
            [results[b].get(s, 0) for b in BATCH_SIZES],
            w,
            label=f"Stage {int(s) + 1}",
        )

    plt.xticks(x + w * (len(stages) - 1) / 2, BATCH_SIZES)
    plt.xlabel("Batch Size")
    plt.ylabel("Execution Time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


teacher_results = {}
student_results = {}

for bsz in BATCH_SIZES:
    df = load_csv(
        BASE_DIR / f"dpipe-profile-b{bsz}" / f"dpipe-profile-b{bsz}" / CSV_NAME
    )

    teacher_results[bsz] = stage_time_sum(df, "teacher")
    student_results[bsz] = stage_time_sum(df, "student")


plot(
    teacher_results,
    "Per-stage execution time across batch sizes (ViT-Large, DPipe)",
    "figure10_dpipe_vit_large_teacher.png",
)

plot(
    student_results,
    "Per-stage execution time across batch sizes (ResNet-152, DPipe)",
    "figure11_dpipe_resnet152_student.png",
)
