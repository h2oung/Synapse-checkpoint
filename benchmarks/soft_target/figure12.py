import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path("results/st")
BATCH_SIZES = [8, 16, 32, 64, 128]
CSV_NAME = "tspipe_profile.csv"
TARGET_ITER = 7


def load_csv(path):
    return pd.read_csv(
        path,
        header=None,
        names=["ts", "batch", "view", "ubatch", "is_target", "src", "dst", "op"],
    )


def normalize_is_target(x):
    if isinstance(x, str):
        return x.upper() == "TRUE"
    return bool(x)


def per_stage_time(df):
    start = df[~df.op.str.endswith("_finish")].copy()
    end = df[df.op.str.endswith("_finish")].copy()

    start["op_base"] = start.op
    end["op_base"] = end.op.str.replace("_finish", "", regex=False)

    merged = start.merge(
        end,
        on=["batch", "view", "ubatch", "dst", "op_base"],
        suffixes=("_s", "_e"),
    )

    merged["dur"] = merged.ts_e - merged.ts_s
    return merged.groupby("dst")["dur"].sum()


def iteration_latency(csv_path):
    df = load_csv(csv_path)
    df["is_target_norm"] = df["is_target"].apply(normalize_is_target)

    if TARGET_ITER in df.batch.unique():
        target_batch = TARGET_ITER
    else:
        target_batch = sorted(df.batch.unique())[0]

    df = df[df.batch == target_batch]

    stage_times = per_stage_time(df)

    if len(stage_times) == 0:
        return None

    return stage_times.max()


tspipe_latency = []
dpipe_latency = []

for bsz in BATCH_SIZES:
    tspipe_csv = BASE_DIR / f"baseline-profile-b{bsz}" / f"baseline-profile-b{bsz}" / CSV_NAME
    dpipe_csv = BASE_DIR / f"dpipe-profile-b{bsz}" / f"dpipe-profile-b{bsz}" / CSV_NAME

    tspipe_latency.append(iteration_latency(tspipe_csv))
    dpipe_latency.append(iteration_latency(dpipe_csv))


plt.figure(figsize=(8, 4))

x = np.arange(len(BATCH_SIZES))
w = 0.35

plt.bar(x - w/2, tspipe_latency, width=w, label="TSPipe")
plt.bar(x + w/2, dpipe_latency, width=w, label="TSPipe + Planner")

plt.xticks(x, BATCH_SIZES)
plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.title("Per-iteration latency comparison")
plt.legend()
plt.tight_layout()
plt.savefig("figure12_iteration_latency_comparison.png")
plt.close()
