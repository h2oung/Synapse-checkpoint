#!/usr/bin/env python3
"""Create an onset-aligned, normalized comparison figure for GPU-3 background load."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}
WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")


@dataclass
class RunSeries:
    label: str
    run_dir: Path
    total_seconds: int
    restart_count: int
    steps: np.ndarray
    values_ms: np.ndarray
    smooth_ms: np.ndarray
    onset_step: Optional[int]
    wall_detected_step: Optional[int]
    replan_step: Optional[int]
    reeval_step: Optional[int]
    baseline_ms: float
    aligned_steps: np.ndarray
    normalized_ratio: np.ndarray
    color: str


def rolling_mean(values: Sequence[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1 or window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rolling_median(values: Sequence[float], window: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1 or window <= 1:
        return arr.copy()
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.asarray([np.median(padded[idx : idx + window]) for idx in range(arr.size)], dtype=float)


def smooth_series(values: Sequence[float]) -> np.ndarray:
    # Median-first smoothing reduces one-step spikes while preserving the replan transition.
    return rolling_mean(rolling_median(values, 9), 5)


def parse_summary(path: Path) -> Tuple[int, int]:
    total_seconds = None
    restart_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Total wall-clock time:" in line:
            match = re.search(r"Total wall-clock time:\s+(\d+)s", line)
            if match:
                total_seconds = int(match.group(1))
        elif "Restart count:" in line:
            match = re.search(r"Restart count:\s+(\d+)", line)
            if match:
                restart_count = int(match.group(1))
    if total_seconds is None:
        raise ValueError(f"Could not parse total time from {path}")
    return total_seconds, restart_count


def parse_events(log_path: Path) -> Dict[str, Optional[int]]:
    events: Dict[str, Optional[int]] = {
        "wall_detected": None,
        "replan": None,
        "reeval": None,
    }
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if events["wall_detected"] is None:
            match = WALL_DETECTED_RE.search(line)
            if match:
                events["wall_detected"] = int(match.group(1))
        if events["reeval"] is None:
            match = REEVAL_RE.search(line)
            if match:
                events["reeval"] = int(match.group(1))
        match = POLICY_RE.search(line)
        if match and match.group(1) == "REPLAN":
            events["replan"] = int(match.group(2))
    return events


def infer_onset(steps: np.ndarray, values: np.ndarray, threshold_ratio: float = 1.10) -> Optional[int]:
    if values.size < 8:
        return None
    baseline = float(np.mean(values[: min(30, values.size)]))
    smooth = smooth_series(values)
    threshold = baseline * threshold_ratio
    for idx, step in enumerate(steps):
        if step < 20:
            continue
        end = min(idx + 3, smooth.size)
        if end - idx < 3:
            break
        if np.all(smooth[idx:end] > threshold):
            return int(step)
    return None


def load_partition3_compute(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    profiling_dir = run_dir / "profiling_logs"
    values = defaultdict(float)
    for path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if bool(row.get("target")):
                    continue
                if int(row.get("partition", -1)) != 3:
                    continue
                if row.get("task_name") not in COMPUTE_TASKS:
                    continue
                global_step = row.get("global_step")
                if global_step is None:
                    continue
                values[int(global_step)] += float(row.get("exec_wall_ms", row.get("time_ms", 0.0)) or 0.0)
    steps = np.asarray(sorted(values.keys()), dtype=int)
    series = np.asarray([values[int(step)] for step in steps], dtype=float)
    return steps, series


def build_run(run_dir: Path, label: str, color: str) -> RunSeries:
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    events = parse_events(run_dir / "log.txt")
    steps, values_ms = load_partition3_compute(run_dir)
    smooth_ms = smooth_series(values_ms)
    onset_step = infer_onset(steps, values_ms)
    if onset_step is None:
        onset_step = int(steps[min(20, max(0, len(steps) - 1))]) if len(steps) else 0

    baseline_mask = steps < onset_step
    if np.any(baseline_mask):
        baseline_values = values_ms[baseline_mask]
    else:
        baseline_values = values_ms[: min(30, len(values_ms))]
    if baseline_values.size == 0:
        baseline_ms = 1.0
    else:
        baseline_ms = float(np.mean(baseline_values[: min(30, baseline_values.size)]))
        if baseline_ms <= 0:
            baseline_ms = 1.0

    aligned_steps = steps - int(onset_step)
    normalized_ratio = smooth_ms / baseline_ms

    return RunSeries(
        label=label,
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        steps=steps,
        values_ms=values_ms,
        smooth_ms=smooth_ms,
        onset_step=onset_step,
        wall_detected_step=events.get("wall_detected"),
        replan_step=events.get("replan"),
        reeval_step=events.get("reeval"),
        baseline_ms=baseline_ms,
        aligned_steps=aligned_steps,
        normalized_ratio=normalized_ratio,
        color=color,
    )


def relative_step(run: RunSeries, step: Optional[int]) -> Optional[int]:
    if step is None or run.onset_step is None:
        return None
    return int(step) - int(run.onset_step)


def save_csv(path: Path, runs: Iterable[RunSeries]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "total_seconds",
                "restart_count",
                "baseline_ms",
                "onset_step",
                "wall_detected_step",
                "wall_detected_relative",
                "replan_step",
                "replan_relative",
                "reeval_step",
                "reeval_relative",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "total_seconds": run.total_seconds,
                    "restart_count": run.restart_count,
                    "baseline_ms": f"{run.baseline_ms:.4f}",
                    "onset_step": run.onset_step,
                    "wall_detected_step": run.wall_detected_step,
                    "wall_detected_relative": relative_step(run, run.wall_detected_step),
                    "replan_step": run.replan_step,
                    "replan_relative": relative_step(run, run.replan_step),
                    "reeval_step": run.reeval_step,
                    "reeval_relative": relative_step(run, run.reeval_step),
                }
            )


def plot_aligned(tspipe: RunSeries, failover: RunSeries, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
        }
    )

    rel_points = [0]
    for step in [failover.replan_step, failover.reeval_step, failover.wall_detected_step, tspipe.wall_detected_step]:
        rel = relative_step(failover if step in {failover.replan_step, failover.reeval_step, failover.wall_detected_step} else tspipe, step)
        if rel is not None:
            rel_points.append(rel)
    x_min = min(-8, min(rel_points) - 6)
    x_max = max(55, max(rel_points) + 12)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, 7.9),
        gridspec_kw={"height_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    ax = axes[0]
    for run in [tspipe, failover]:
        mask = (run.aligned_steps >= x_min) & (run.aligned_steps <= x_max)
        ax.plot(run.aligned_steps[mask], run.normalized_ratio[mask], color=run.color, linewidth=3.0, label=run.label)

    ax.axvline(0, color="#E67E22", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.text(0.5, 2.13, "Steps from overload (after 120s)", ha="left", va="top", fontsize=10, color="#E67E22")

    replan_rel = relative_step(failover, failover.replan_step)
    reeval_rel = relative_step(failover, failover.reeval_step)
    if replan_rel is not None:
        ax.axvline(replan_rel, color="#059669", linestyle="-", linewidth=2.0)
        ax.text(replan_rel + 1.0, 1.92, "REPLAN", ha="left", va="top", fontsize=11, color="#059669", fontweight="bold")
    if replan_rel is not None and reeval_rel is not None:
        ax.axvspan(replan_rel, reeval_rel, color="#F3E3B5", alpha=0.45)

    ax.set_title("Compute time")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Norm. compute time")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.85, max(2.15, float(np.nanmax(failover.normalized_ratio[(failover.aligned_steps >= x_min) & (failover.aligned_steps <= x_max)])) + 0.15))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), frameon=False)

    ax = axes[1]
    runs = [tspipe, failover]
    bars = ax.bar([run.label for run in runs], [run.total_seconds / 60.0 for run in runs], color=[run.color for run in runs], width=0.56, edgecolor="#2F2A24", linewidth=1.0)
    for bar, run in zip(bars, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{run.total_seconds / 60.0:.1f}m",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2F2A24",
        )
    ax.set_title("Completion time")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, max(run.total_seconds / 60.0 for run in runs) + 3.0)


    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_base.with_suffix(".csv")
    save_csv(csv_path, runs)

    print(png_path)
    print(pdf_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tspipe-run", required=True, type=Path)
    parser.add_argument("--failover-run", required=True, type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/workspace/Synapse/Synapse/results/figures/bgload_gpu3_b128_onset_aligned_compare"),
    )
    args = parser.parse_args()

    tspipe = build_run(args.tspipe_run, "TSPipe baseline", "#4C6A92")
    failover = build_run(args.failover_run, "Failover + REPLAN", "#C46A2D")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_aligned(tspipe, failover, args.output_base)


if __name__ == "__main__":
    main()
