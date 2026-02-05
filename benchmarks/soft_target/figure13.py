import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("results/st")
BATCH_SIZES = [8, 16, 32, 64, 128]
CSV_NAME = "tspipe_profile.csv"

WARMUP_BATCH = 4

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

def compute_iter_util(df, batch_id):
    df = df[df.batch == batch_id].copy()
    if df.empty:
        return None

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

    compute_rows = merged[merged.op_base == "compute"]

    active_time = compute_rows.groupby("dst")["dur"].sum().mean()
    total_time = df.ts.max() - df.ts.min()

    if total_time <= 0:
        return None

    return active_time / total_time * 100

def calc_avg_gpu_util(df):
    valid_batches = sorted(b for b in df.batch.unique() if b >= WARMUP_BATCH)
    utils = []

    for b in valid_batches:
        u = compute_iter_util(df, b)
        if u is not None:
            utils.append(u)

    if not utils:
        return 0.0

    return float(np.mean(utils))

tspipe_utils = []
dpipe_utils = []

for bsz in BATCH_SIZES:
    tspipe_csv = BASE_DIR / f"baseline-profile-b{bsz}" / f"baseline-profile-b{bsz}" / CSV_NAME
    dpipe_csv = BASE_DIR / f"dpipe-profile-b{bsz}" / f"dpipe-profile-b{bsz}" / CSV_NAME

    tspipe_df = load_csv(tspipe_csv)
    dpipe_df = load_csv(dpipe_csv)

    tspipe_utils.append(calc_avg_gpu_util(tspipe_df))
    dpipe_utils.append(calc_avg_gpu_util(dpipe_df))

x = np.arange(len(BATCH_SIZES))
w = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - w/2, tspipe_utils, width=w, label="TSPipe", color="gray")
plt.bar(x + w/2, dpipe_utils, width=w, label="TSPipe + Planner", color="orangered")

plt.xticks(x, BATCH_SIZES)
plt.xlabel("Batch Size")
plt.ylabel("GPU Utilization (%)")
plt.title("Figure 13: Average GPU Utilization Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("figure13_gpu_utilization_comparison.png")
plt.show()
