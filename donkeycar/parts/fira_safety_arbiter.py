import time
from typing import Optional, Tuple

import numpy as np


class FiraSafetyArbiter:
    """
    Lightweight safety arbiter for FIRA modes.

    The arbiter is intentionally conservative:
    - clamps steering and throttle to configured limits,
    - smoothly limits command rate changes,
    - applies curve-speed reduction from steering magnitude,
    - blends optional lane guidance when confidence is high enough,
    - applies optional obstacle steering correction and speed reduction,
    - applies detector-failsafe throttle cap when a configured detector is down.
    """

    def __init__(
        self,
        max_abs_steering: float = 1.0,
        throttle_min: float = 0.0,
        throttle_max: float = 1.0,
        curve_throttle_start: float = 0.25,
        curve_throttle_full: float = 0.9,
        curve_throttle_factor_min: float = 0.6,
        failsafe_throttle_cap: float = 0.08,
        severe_failsafe_throttle_cap: float = 0.04,
        use_yolo_detector_failsafe: bool = False,
        use_tf_detector_failsafe: bool = False,
        max_steering_delta_per_sec: float = 2.0,
        max_throttle_delta_per_sec: float = 0.8,
        use_lane_guidance: bool = False,
        lane_confidence_min: float = 0.35,
        lane_blend_max: float = 0.7,
        lane_angle_max_abs: float = 1.0,
        use_obstacle_correction: bool = False,
        obstacle_correction_max_abs: float = 1.0,
        obstacle_blend_max: float = 1.0,
        obstacle_throttle_factor_min: float = 0.45,
        debug: bool = False,
    ) -> None:
        self.max_abs_steering = float(max_abs_steering)
        self.throttle_min = float(throttle_min)
        self.throttle_max = float(throttle_max)

        self.curve_throttle_start = float(curve_throttle_start)
        self.curve_throttle_full = float(curve_throttle_full)
        self.curve_throttle_factor_min = float(curve_throttle_factor_min)

        self.failsafe_throttle_cap = float(failsafe_throttle_cap)
        self.severe_failsafe_throttle_cap = float(severe_failsafe_throttle_cap)
        self.use_yolo_detector_failsafe = bool(use_yolo_detector_failsafe)
        self.use_tf_detector_failsafe = bool(use_tf_detector_failsafe)

        self.max_steering_delta_per_sec = float(max_steering_delta_per_sec)
        self.max_throttle_delta_per_sec = float(max_throttle_delta_per_sec)

        self.use_lane_guidance = bool(use_lane_guidance)
        self.lane_confidence_min = float(lane_confidence_min)
        self.lane_blend_max = float(lane_blend_max)
        self.lane_angle_max_abs = float(lane_angle_max_abs)

        self.use_obstacle_correction = bool(use_obstacle_correction)
        self.obstacle_correction_max_abs = float(obstacle_correction_max_abs)
        self.obstacle_blend_max = float(obstacle_blend_max)
        self.obstacle_throttle_factor_min = float(obstacle_throttle_factor_min)

        self.debug = bool(debug)

        self._last_angle: Optional[float] = None
        self._last_throttle: Optional[float] = None
        self._last_time: Optional[float] = None

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _to_float(value, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_detector_down(enabled: bool, ready_value: Optional[object]) -> bool:
        if not enabled:
            return False
        # Missing telemetry defaults to safe mode to avoid blind driving.
        if ready_value is None:
            return True
        return not bool(ready_value)

    @staticmethod
    def _health_ready_and_severity(ready_value: Optional[object], health: Optional[dict]) -> Tuple[bool, int]:
        severity = 0
        ready = ready_value

        if isinstance(health, dict):
            if ready is None:
                ready = health.get('detector_ready', None)

            backend = str(health.get('detector_backend', 'unavailable')).lower()
            last_error = str(health.get('last_detection_error', '') or '').lower()
            if backend == 'unavailable' or 'backend unavailable' in last_error:
                severity = 2
            elif last_error:
                severity = 1

        if ready is None:
            return False, max(1, severity)

        return bool(ready), severity

    def _curve_factor(self, angle_abs: float) -> float:
        if angle_abs <= self.curve_throttle_start:
            return 1.0

        if angle_abs >= self.curve_throttle_full:
            return self.curve_throttle_factor_min

        span = max(1e-6, self.curve_throttle_full - self.curve_throttle_start)
        alpha = (angle_abs - self.curve_throttle_start) / span
        return 1.0 - alpha * (1.0 - self.curve_throttle_factor_min)

    @staticmethod
    def _slew_limit(target: float, previous: float, max_delta: float) -> float:
        if target > previous + max_delta:
            return previous + max_delta
        if target < previous - max_delta:
            return previous - max_delta
        return target

    def _lane_blend_weight(self, lane_confidence: Optional[float]) -> float:
        if lane_confidence is None:
            return 0.0

        confidence = self._clip(lane_confidence, 0.0, 1.0)
        if confidence <= self.lane_confidence_min:
            return 0.0

        span = max(1e-6, 1.0 - self.lane_confidence_min)
        alpha = (confidence - self.lane_confidence_min) / span
        return self._clip(alpha * self.lane_blend_max, 0.0, self.lane_blend_max)

    def run(
        self,
        angle: float,
        throttle: float,
        img_arr,
        yolo_detector_ready=None,
        tf_detector_ready=None,
        yolo_detector_health=None,
        tf_detector_health=None,
        lane_angle=None,
        lane_confidence=None,
        obstacle_steering_correction=None,
        obstacle_severity=None,
        drive_state=None,
        drive_state_throttle_cap=None,
    ) -> Tuple[float, float, object]:
        now = time.time()
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)

        # Baseline hard limits.
        safe_angle = self._clip(float(angle), -self.max_abs_steering, self.max_abs_steering)
        safe_throttle = self._clip(float(throttle), self.throttle_min, self.throttle_max)

        # Lane guidance blend (pilot -> lane), gated by confidence.
        lane_target = self._to_float(lane_angle)
        lane_conf = self._to_float(lane_confidence)
        lane_weight = 0.0
        if self.use_lane_guidance and lane_target is not None:
            lane_target = self._clip(lane_target, -self.lane_angle_max_abs, self.lane_angle_max_abs)
            lane_weight = self._lane_blend_weight(lane_conf)
            safe_angle = (1.0 - lane_weight) * safe_angle + lane_weight * lane_target

        # Obstacle correction is applied after lane blending to prioritize avoidance.
        obstacle_corr = self._to_float(obstacle_steering_correction)
        obstacle_level = self._to_float(obstacle_severity, default=1.0)
        obstacle_weight = 0.0
        if self.use_obstacle_correction and obstacle_corr is not None:
            obstacle_level = self._clip(obstacle_level, 0.0, 1.0)
            obstacle_corr = self._clip(
                obstacle_corr,
                -self.obstacle_correction_max_abs,
                self.obstacle_correction_max_abs,
            )
            obstacle_weight = obstacle_level * self._clip(self.obstacle_blend_max, 0.0, 1.0)
            safe_angle += obstacle_corr * obstacle_weight

            obstacle_throttle_factor_min = self._clip(self.obstacle_throttle_factor_min, 0.0, 1.0)
            obstacle_speed_factor = 1.0 - obstacle_level * (1.0 - obstacle_throttle_factor_min)
            safe_throttle *= self._clip(obstacle_speed_factor, obstacle_throttle_factor_min, 1.0)

        # Speed adaptation from steering demand.
        curve_factor = self._curve_factor(abs(safe_angle))
        safe_throttle *= curve_factor

        # Detector status failsafe.
        yolo_ready, yolo_severity = self._health_ready_and_severity(yolo_detector_ready, yolo_detector_health)
        tf_ready, tf_severity = self._health_ready_and_severity(tf_detector_ready, tf_detector_health)

        yolo_down = self._is_detector_down(self.use_yolo_detector_failsafe, yolo_ready)
        tf_down = self._is_detector_down(self.use_tf_detector_failsafe, tf_ready)
        if yolo_down or tf_down:
            throttle_cap = self.failsafe_throttle_cap
            if yolo_severity >= 2 or tf_severity >= 2:
                throttle_cap = min(throttle_cap, self.severe_failsafe_throttle_cap)
            safe_throttle = min(safe_throttle, throttle_cap)

        state_throttle_cap = self._to_float(drive_state_throttle_cap)
        if state_throttle_cap is not None:
            safe_throttle = min(
                safe_throttle,
                self._clip(state_throttle_cap, self.throttle_min, self.throttle_max),
            )

        # Temporal smoothing.
        if self._last_angle is not None and dt > 0.0:
            safe_angle = self._slew_limit(
                safe_angle,
                self._last_angle,
                self.max_steering_delta_per_sec * dt,
            )

        if self._last_throttle is not None and dt > 0.0:
            safe_throttle = self._slew_limit(
                safe_throttle,
                self._last_throttle,
                self.max_throttle_delta_per_sec * dt,
            )

        safe_angle = self._clip(safe_angle, -self.max_abs_steering, self.max_abs_steering)
        safe_throttle = self._clip(safe_throttle, self.throttle_min, self.throttle_max)

        self._last_angle = safe_angle
        self._last_throttle = safe_throttle
        self._last_time = now

        if self.debug and (yolo_down or tf_down):
            print(
                "[FIRA SAFETY] detector failsafe active | "
                f"yolo_ready={yolo_ready} tf_ready={tf_ready} "
                f"throttle={safe_throttle:.3f}"
            )

        if self.debug and (lane_weight > 0.0 or obstacle_weight > 0.0):
            print(
                "[FIRA SAFETY] fusion | "
                f"lane_w={lane_weight:.2f} obstacle_w={obstacle_weight:.2f} "
                f"angle={safe_angle:.3f} throttle={safe_throttle:.3f}"
            )

        if self.debug and drive_state is not None:
            print(
                "[FIRA SAFETY] drive state | "
                f"state={drive_state} cap={state_throttle_cap} throttle={safe_throttle:.3f}"
            )

        return safe_angle, safe_throttle, img_arr


class FiraSafetyArbiterTelemetryAdapter:
    """Wrap FiraSafetyArbiter and expose additional telemetry channels."""

    def __init__(self, arbiter: FiraSafetyArbiter):
        self.arbiter = arbiter

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def run(
        self,
        angle: float,
        throttle: float,
        img_arr,
        yolo_detector_health=None,
        tf_detector_health=None,
        lane_angle=None,
        lane_confidence=None,
        obstacle_steering_correction=None,
        obstacle_severity=None,
        drive_state=None,
        drive_state_throttle_cap=None,
    ):
        safe_angle, safe_throttle, out_img = self.arbiter.run(
            angle=angle,
            throttle=throttle,
            img_arr=img_arr,
            yolo_detector_health=yolo_detector_health,
            tf_detector_health=tf_detector_health,
            lane_angle=lane_angle,
            lane_confidence=lane_confidence,
            obstacle_steering_correction=obstacle_steering_correction,
            obstacle_severity=obstacle_severity,
            drive_state=drive_state,
            drive_state_throttle_cap=drive_state_throttle_cap,
        )

        yolo_ready, _ = self.arbiter._health_ready_and_severity(None, yolo_detector_health)
        tf_ready, _ = self.arbiter._health_ready_and_severity(None, tf_detector_health)
        yolo_down = self.arbiter._is_detector_down(self.arbiter.use_yolo_detector_failsafe, yolo_ready)
        tf_down = self.arbiter._is_detector_down(self.arbiter.use_tf_detector_failsafe, tf_ready)
        detector_failsafe_active = bool(yolo_down or tf_down)

        lane_conf = self.arbiter._clip(self._to_float(lane_confidence, 0.0), 0.0, 1.0)
        lane_weight = self.arbiter._lane_blend_weight(lane_conf) if self.arbiter.use_lane_guidance else 0.0

        obstacle_level = self.arbiter._clip(self._to_float(obstacle_severity, 0.0), 0.0, 1.0)
        obstacle_weight = 0.0
        if self.arbiter.use_obstacle_correction:
            obstacle_weight = obstacle_level * self.arbiter._clip(self.arbiter.obstacle_blend_max, 0.0, 1.0)

        curve_factor = self.arbiter._curve_factor(abs(safe_angle))
        speed_limit_factor = 1.0
        requested = max(1e-6, abs(float(throttle)))
        if requested > 0.0:
            speed_limit_factor = self.arbiter._clip(abs(safe_throttle) / requested, 0.0, 1.0)

        return (
            safe_angle,
            safe_throttle,
            out_img,
            detector_failsafe_active,
            float(lane_weight),
            float(obstacle_weight),
            float(curve_factor),
            float(speed_limit_factor),
        )
