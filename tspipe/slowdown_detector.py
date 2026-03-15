"""
Slowdown 탐지 및 모니터링 시스템

GPU 성능 저하를 baseline과 비교하여 감지하고,
필요시 α_comp, β_comm으로 분해

사용:
  detector = SlowdownDetector()
  
  for batch in training:
    stage_time = measure_stage_time()
    detector.record_stage_time(stage_time)
    
    slowdown = detector.get_slowdown_ratio()
    if slowdown > 1.05:
      α, β = alpha_beta_loader.get_alpha_beta(slowdown)
      policy = decide_policy(α, β)
"""

import numpy as np
from collections import deque
from typing import Optional, Dict
import logging


class SlowdownDetector:
    """Stage time 기반 GPU slowdown 탐지"""
    
    def __init__(self, 
                 baseline_window: int = 20,
                 detection_window: int = 10,
                 slowdown_threshold: float = 1.05):
        """
        Args:
            baseline_window: Baseline 설정할 초기 배치 수 (첫 20개)
            detection_window: Slowdown 판단할 최근 배치 수 (최근 10개)
            slowdown_threshold: Slowdown 감지 임계값 (5% 이상)
        """
        self.logger = logging.getLogger(f"{__name__}.SlowdownDetector")
        
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.slowdown_threshold = slowdown_threshold
        
        # Stage time 저장 (최대 detection_window개 유지)
        self.stage_times = deque(maxlen=max(baseline_window, detection_window))
        
        # Baseline (정상 상태)
        self.baseline_stage_time: Optional[float] = None
        self.baseline_std: Optional[float] = None
        
        # 배치 개수 추적
        self.batch_count = 0
        self.slowdown_detected_at_step = None
        
        self.logger.info("✅ SlowdownDetector initialized")
    
    def record_stage_time(self, stage_time_ms: float):
        """
        매 배치마다 호출
        
        Args:
            stage_time_ms: 이번 배치의 max stage time (ms단위)
        """
        self.stage_times.append(stage_time_ms)
        self.batch_count += 1
        
        # 초기 baseline 설정 (첫 baseline_window개 배치)
        if self.baseline_stage_time is None and self.batch_count >= self.baseline_window:
            first_batches = list(self.stage_times)[:self.baseline_window]
            self.baseline_stage_time = np.mean(first_batches)
            self.baseline_std = np.std(first_batches)
            
            self.logger.info(
                f"📊 Baseline set: {self.baseline_stage_time:.2f}ms "
                f"(±{self.baseline_std:.2f}ms)"
            )
    
    def get_slowdown_ratio(self) -> float:
        """
        현재 slowdown ratio 계산
        
        Returns:
            slowdown: 1.0 = 정상, 1.2 = 20% 느려짐
        """
        
        if self.baseline_stage_time is None:
            return 1.0
        
        # 최근 detection_window개 배치의 평균
        if len(self.stage_times) >= self.detection_window:
            recent = list(self.stage_times)[-self.detection_window:]
            avg_recent = np.mean(recent)
            slowdown = avg_recent / self.baseline_stage_time
            return slowdown
        
        return 1.0
    
    def is_slowdown_detected(self, threshold: Optional[float] = None) -> bool:
        """
        Slowdown 탐지 여부
        
        Args:
            threshold: 임계값 (기본: self.slowdown_threshold)
        
        Returns:
            bool: Slowdown이 감지되었는가?
        """
        if threshold is None:
            threshold = self.slowdown_threshold
        
        slowdown = self.get_slowdown_ratio()
        detected = slowdown > threshold
        
        if detected and self.slowdown_detected_at_step is None:
            self.slowdown_detected_at_step = self.batch_count
            self.logger.warning(
                f"⚠️ Slowdown detected at step {self.batch_count}: "
                f"{slowdown:.3f}x (threshold: {threshold:.2f})"
            )
        
        return detected
    
    def get_statistics(self) -> Dict:
        """현재 통계 정보 반환"""
        
        if self.baseline_stage_time is None:
            return {
                'status': 'baseline_not_set',
                'batch_count': self.batch_count,
                'baseline_required': self.baseline_window - self.batch_count
            }
        
        slowdown = self.get_slowdown_ratio()
        recent = list(self.stage_times)[-self.detection_window:] if self.stage_times else []
        
        return {
            'status': 'normal' if not self.is_slowdown_detected() else 'slowdown_detected',
            'batch_count': self.batch_count,
            'baseline_stage_time_ms': self.baseline_stage_time,
            'baseline_std_ms': self.baseline_std,
            'current_slowdown_ratio': slowdown,
            'recent_avg_ms': np.mean(recent) if recent else None,
            'recent_min_ms': np.min(recent) if recent else None,
            'recent_max_ms': np.max(recent) if recent else None,
            'slowdown_detected_at_step': self.slowdown_detected_at_step
        }
