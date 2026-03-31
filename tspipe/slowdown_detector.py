import time
import numpy as np
from collections import deque
from typing import Optional, Dict
import logging


class SlowdownDetector:
    """Stage time 기반 wall-clock slowdown 탐지 및 sustained trigger 관리."""

    def __init__(
        self,
        baseline_window: int = 10,
        detection_window: int = 5,
        slowdown_threshold: float = 1.10,
        inject_scenario: str = "",
    ):
        self.logger = logging.getLogger(f"{__name__}.SlowdownDetector")

        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.slowdown_threshold = slowdown_threshold
        self.inject_scenario = inject_scenario.strip()

        self.stage_times = deque(maxlen=max(baseline_window, detection_window))

        self.baseline_stage_time: Optional[float] = None
        self.baseline_std: Optional[float] = None

        self.batch_count = 0
        self.slowdown_detected_at_step = None

        # wall-clock sustained trigger state
        self.wallclock_slowdown_started_at: Optional[float] = None
        self.wallclock_sustained_duration_sec: float = 0.0

        self.logger.info("✅ SlowdownDetector initialized")

    def record_stage_time(self, stage_time_ms: float, timestamp_sec: Optional[float] = None):
        """
        매 step마다 wall-clock elapsed time(ms)를 기록한다.
        """
        self.stage_times.append(float(stage_time_ms))
        self.batch_count += 1

        now = float(timestamp_sec) if timestamp_sec is not None else time.time()

        # baseline 확정
        if self.baseline_stage_time is None and self.batch_count >= self.baseline_window:
            first_batches = list(self.stage_times)[:self.baseline_window]
            self.baseline_stage_time = float(np.mean(first_batches))
            self.baseline_std = float(np.std(first_batches))
            self.logger.info(
                f"📊 Baseline set: {self.baseline_stage_time:.2f}ms "
                f"(±{self.baseline_std:.2f}ms)"
            )

        if self.baseline_stage_time is not None:
            self._update_wallclock_trigger_state(now)

    def _update_wallclock_trigger_state(self, now: float):
        slowdown = self.get_slowdown_ratio()

        if slowdown > self.slowdown_threshold:
            if self.wallclock_slowdown_started_at is None:
                self.wallclock_slowdown_started_at = now
                self.wallclock_sustained_duration_sec = 0.0
            else:
                self.wallclock_sustained_duration_sec = max(
                    0.0, now - self.wallclock_slowdown_started_at
                )

            if self.slowdown_detected_at_step is None:
                self.slowdown_detected_at_step = self.batch_count
                self.logger.warning(
                    f"⚠️ Wall-clock slowdown detected at step {self.batch_count}: "
                    f"{slowdown:.3f}x (threshold: {self.slowdown_threshold:.2f})"
                )
        else:
            self.wallclock_slowdown_started_at = None
            self.wallclock_sustained_duration_sec = 0.0
            self.slowdown_detected_at_step = None

    def get_slowdown_ratio(self) -> float:
        if self.baseline_stage_time is None:
            return 1.0

        if len(self.stage_times) >= self.detection_window:
            recent = list(self.stage_times)[-self.detection_window:]
            avg_recent = float(np.mean(recent))
            slowdown = avg_recent / self.baseline_stage_time
            return slowdown

        return 1.0

    def is_slowdown_detected(self, threshold: Optional[float] = None) -> bool:
        threshold = self.slowdown_threshold if threshold is None else float(threshold)
        return self.get_slowdown_ratio() > threshold

    def get_trigger_state(self, sustain_sec: float) -> Dict:
        slowdown = self.get_slowdown_ratio()
        triggered = (
            slowdown > self.slowdown_threshold
            and self.wallclock_sustained_duration_sec >= float(sustain_sec)
        )
        return {
            "current_slowdown_ratio": slowdown,
            "sustained_duration_sec": self.wallclock_sustained_duration_sec,
            "triggered": triggered,
            "threshold": self.slowdown_threshold,
            "sustain_sec_required": float(sustain_sec),
            "baseline_stage_time_ms": self.baseline_stage_time,
            "baseline_std_ms": self.baseline_std,
            "batch_count": self.batch_count,
        }

    def get_statistics(self) -> Dict:
        if self.baseline_stage_time is None:
            return {
                "status": "baseline_not_set",
                "batch_count": self.batch_count,
                "baseline_required": self.baseline_window - self.batch_count,
            }

        slowdown = self.get_slowdown_ratio()
        recent = list(self.stage_times)[-self.detection_window:] if self.stage_times else []

        return {
            "status": "normal" if not self.is_slowdown_detected() else "slowdown_detected",
            "batch_count": self.batch_count,
            "baseline_stage_time_ms": self.baseline_stage_time,
            "baseline_std_ms": self.baseline_std,
            "current_slowdown_ratio": slowdown,
            "recent_avg_ms": float(np.mean(recent)) if recent else None,
            "recent_min_ms": float(np.min(recent)) if recent else None,
            "recent_max_ms": float(np.max(recent)) if recent else None,
            "wallclock_sustained_duration_sec": self.wallclock_sustained_duration_sec,
            "slowdown_detected_at_step": self.slowdown_detected_at_step,
        }