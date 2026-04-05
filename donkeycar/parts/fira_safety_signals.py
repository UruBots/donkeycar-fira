from typing import Optional, Tuple

import cv2
import numpy as np


class FiraSafetySignalsEstimator:
    """Estimate lane and obstacle signals from camera frames for safety fusion."""

    def __init__(
        self,
        lane_enabled: bool = True,
        lane_roi_top_ratio: float = 0.45,
        lane_color_low: Tuple[int, int, int] = (18, 70, 70),
        lane_color_high: Tuple[int, int, int] = (42, 255, 255),
        lane_min_pixel_ratio: float = 0.003,
        lane_full_pixel_ratio: float = 0.04,
        lane_steering_gain: float = 1.0,
        lane_angle_max_abs: float = 1.0,
        obstacle_enabled: bool = True,
        obstacle_roi_top_ratio: float = 0.45,
        obstacle_roi_bottom_ratio: float = 0.95,
        obstacle_center_band_ratio: float = 0.85,
        obstacle_color_low: Tuple[int, int, int] = (5, 90, 90),
        obstacle_color_high: Tuple[int, int, int] = (24, 255, 255),
        obstacle_min_pixel_ratio: float = 0.002,
        obstacle_full_pixel_ratio: float = 0.03,
        obstacle_max_correction: float = 0.9,
        obstacle_lane_change_enabled: bool = True,
        obstacle_engage_severity: float = 0.30,
        obstacle_avoid_min_correction: float = 0.20,
        obstacle_avoid_min_severity: float = 0.35,
        obstacle_clear_frames: int = 2,
        obstacle_recover_frames: int = 6,
        obstacle_recover_max_correction: float = 0.16,
        obstacle_recover_min_severity: float = 0.20,
        reduce_signals_when_detector_down: bool = True,
        detector_down_signal_scale: float = 0.5,
        enable_temporal_smoothing: bool = True,
        smoothing_alpha: float = 0.35,
        debug: bool = False,
    ) -> None:
        self.lane_enabled = bool(lane_enabled)
        self.lane_roi_top_ratio = float(lane_roi_top_ratio)
        self.lane_color_low = np.array(lane_color_low, dtype=np.uint8)
        self.lane_color_high = np.array(lane_color_high, dtype=np.uint8)
        self.lane_min_pixel_ratio = float(lane_min_pixel_ratio)
        self.lane_full_pixel_ratio = float(lane_full_pixel_ratio)
        self.lane_steering_gain = float(lane_steering_gain)
        self.lane_angle_max_abs = float(lane_angle_max_abs)

        self.obstacle_enabled = bool(obstacle_enabled)
        self.obstacle_roi_top_ratio = float(obstacle_roi_top_ratio)
        self.obstacle_roi_bottom_ratio = float(obstacle_roi_bottom_ratio)
        self.obstacle_center_band_ratio = float(obstacle_center_band_ratio)
        self.obstacle_color_low = np.array(obstacle_color_low, dtype=np.uint8)
        self.obstacle_color_high = np.array(obstacle_color_high, dtype=np.uint8)
        self.obstacle_min_pixel_ratio = float(obstacle_min_pixel_ratio)
        self.obstacle_full_pixel_ratio = float(obstacle_full_pixel_ratio)
        self.obstacle_max_correction = float(obstacle_max_correction)
        self.obstacle_lane_change_enabled = bool(obstacle_lane_change_enabled)
        self.obstacle_engage_severity = float(obstacle_engage_severity)
        self.obstacle_avoid_min_correction = float(obstacle_avoid_min_correction)
        self.obstacle_avoid_min_severity = float(obstacle_avoid_min_severity)
        self.obstacle_clear_frames = int(obstacle_clear_frames)
        self.obstacle_recover_frames = int(obstacle_recover_frames)
        self.obstacle_recover_max_correction = float(obstacle_recover_max_correction)
        self.obstacle_recover_min_severity = float(obstacle_recover_min_severity)

        self.reduce_signals_when_detector_down = bool(reduce_signals_when_detector_down)
        self.detector_down_signal_scale = float(detector_down_signal_scale)
        self.enable_temporal_smoothing = bool(enable_temporal_smoothing)
        self.smoothing_alpha = float(smoothing_alpha)
        self.debug = bool(debug)

        self._lane_angle_ema: Optional[float] = None
        self._lane_confidence_ema: Optional[float] = None
        self._obstacle_correction_ema: Optional[float] = None
        self._obstacle_severity_ema: Optional[float] = None
        self._obstacle_phase = 'idle'
        self._obstacle_direction = 0.0
        self._obstacle_missing_frames = 0
        self._obstacle_recover_countdown = 0

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _detector_down(yolo_ready: Optional[object], tf_ready: Optional[object]) -> bool:
        for ready in (yolo_ready, tf_ready):
            if ready is None:
                continue
            if not bool(ready):
                return True
        return False

    def _ema(self, previous: Optional[float], current: float) -> float:
        alpha = self._clip(self.smoothing_alpha, 0.0, 1.0)
        if previous is None:
            return current
        return (1.0 - alpha) * previous + alpha * current

    def _profile_obstacle_policy(self, active_profile: Optional[str]):
        profile = str(active_profile or '').strip().upper()
        if profile == 'RACE':
            return {
                'engage_severity': 0.24,
                'avoid_min_corr': 0.30,
                'avoid_min_sev': 0.44,
                'clear_frames': 2,
                'recover_frames': 7,
                'recover_max_corr': 0.14,
                'recover_min_sev': 0.17,
            }
        if profile == 'TIGHT':
            return {
                'engage_severity': 0.20,
                'avoid_min_corr': 0.34,
                'avoid_min_sev': 0.46,
                'clear_frames': 1,
                'recover_frames': 4,
                'recover_max_corr': 0.22,
                'recover_min_sev': 0.15,
            }
        return {
            'engage_severity': self._clip(self.obstacle_engage_severity, 0.0, 1.0),
            'avoid_min_corr': self._clip(self.obstacle_avoid_min_correction, 0.0, self.obstacle_max_correction),
            'avoid_min_sev': self._clip(self.obstacle_avoid_min_severity, 0.0, 1.0),
            'clear_frames': max(1, int(self.obstacle_clear_frames)),
            'recover_frames': max(1, int(self.obstacle_recover_frames)),
            'recover_max_corr': self._clip(self.obstacle_recover_max_correction, 0.0, self.obstacle_max_correction),
            'recover_min_sev': self._clip(self.obstacle_recover_min_severity, 0.0, 1.0),
        }

    def _lane_signal(self, rgb_img: np.ndarray, pilot_angle: float) -> Tuple[float, float]:
        if not self.lane_enabled:
            return pilot_angle, 0.0

        height, width = rgb_img.shape[:2]
        top = int(self._clip(self.lane_roi_top_ratio, 0.0, 0.98) * height)
        roi = rgb_img[top:, :, :]
        if roi.size == 0:
            return pilot_angle, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lane_color_low, self.lane_color_high)

        total_pixels = float(mask.size)
        active_pixels = float(np.count_nonzero(mask))
        pixel_ratio = 0.0 if total_pixels <= 0.0 else active_pixels / total_pixels

        if pixel_ratio < self.lane_min_pixel_ratio or active_pixels < 8.0:
            return pilot_angle, 0.0

        x_coords = np.where(mask > 0)[1]
        centroid_x = float(np.mean(x_coords)) if x_coords.size > 0 else width / 2.0

        center_x = (width - 1.0) * 0.5
        half_w = max(1.0, center_x)
        error = (centroid_x - center_x) / half_w

        lane_angle = self._clip(error * self.lane_steering_gain, -self.lane_angle_max_abs, self.lane_angle_max_abs)

        span = max(1e-6, self.lane_full_pixel_ratio - self.lane_min_pixel_ratio)
        confidence = self._clip((pixel_ratio - self.lane_min_pixel_ratio) / span, 0.0, 1.0)
        return lane_angle, confidence

    def _obstacle_signal(self, rgb_img: np.ndarray) -> Tuple[float, float]:
        if not self.obstacle_enabled:
            return 0.0, 0.0

        height, width = rgb_img.shape[:2]
        top = int(self._clip(self.obstacle_roi_top_ratio, 0.0, 0.98) * height)
        bottom = int(self._clip(self.obstacle_roi_bottom_ratio, 0.02, 1.0) * height)
        if bottom <= top:
            return 0.0, 0.0

        center_band = self._clip(self.obstacle_center_band_ratio, 0.2, 1.0)
        x_margin = int((1.0 - center_band) * width * 0.5)
        x0 = x_margin
        x1 = width - x_margin
        if x1 <= x0:
            return 0.0, 0.0

        roi = rgb_img[top:bottom, x0:x1, :]
        if roi.size == 0:
            return 0.0, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.obstacle_color_low, self.obstacle_color_high)

        total_pixels = float(mask.size)
        active_pixels = float(np.count_nonzero(mask))
        pixel_ratio = 0.0 if total_pixels <= 0.0 else active_pixels / total_pixels

        if pixel_ratio < self.obstacle_min_pixel_ratio or active_pixels < 10.0:
            return 0.0, 0.0

        x_coords = np.where(mask > 0)[1]
        centroid_x = float(np.mean(x_coords)) if x_coords.size > 0 else (x1 - x0) / 2.0

        center_x = (x1 - x0 - 1.0) * 0.5
        half_w = max(1.0, center_x)
        norm_x = (centroid_x - center_x) / half_w

        span = max(1e-6, self.obstacle_full_pixel_ratio - self.obstacle_min_pixel_ratio)
        severity = self._clip((pixel_ratio - self.obstacle_min_pixel_ratio) / span, 0.0, 1.0)

        # If obstacle is on the right (norm_x > 0), steer left (negative correction).
        correction = self._clip(-norm_x * severity, -1.0, 1.0) * self.obstacle_max_correction
        return correction, severity

    def _obstacle_lane_change_policy(
        self,
        raw_correction: float,
        raw_severity: float,
        active_profile: Optional[str] = None,
    ) -> Tuple[float, float]:
        if not self.obstacle_lane_change_enabled:
            return raw_correction, raw_severity

        profile_policy = self._profile_obstacle_policy(active_profile)
        engage_severity = profile_policy['engage_severity']
        avoid_min_corr = profile_policy['avoid_min_corr']
        avoid_min_sev = profile_policy['avoid_min_sev']
        clear_frames = profile_policy['clear_frames']
        recover_frames = profile_policy['recover_frames']
        recover_max_corr = profile_policy['recover_max_corr']
        recover_min_sev = profile_policy['recover_min_sev']

        obstacle_seen = raw_severity >= engage_severity

        if self._obstacle_phase == 'idle':
            if obstacle_seen:
                self._obstacle_phase = 'avoid'
                self._obstacle_direction = -1.0 if raw_correction < 0.0 else 1.0
                self._obstacle_missing_frames = 0

        if self._obstacle_phase == 'avoid':
            if obstacle_seen:
                self._obstacle_missing_frames = 0
                if abs(raw_correction) > 1e-6:
                    self._obstacle_direction = -1.0 if raw_correction < 0.0 else 1.0
            else:
                self._obstacle_missing_frames += 1
                if self._obstacle_missing_frames >= clear_frames:
                    self._obstacle_phase = 'recover'
                    self._obstacle_recover_countdown = recover_frames

            if self._obstacle_phase == 'avoid':
                direction = self._obstacle_direction if abs(self._obstacle_direction) > 0.0 else (-1.0 if raw_correction < 0.0 else 1.0)
                correction = direction * max(abs(raw_correction), avoid_min_corr)
                severity = max(raw_severity, avoid_min_sev)
                return correction, severity

        if self._obstacle_phase == 'recover':
            if obstacle_seen:
                self._obstacle_phase = 'avoid'
                self._obstacle_direction = -1.0 if raw_correction < 0.0 else 1.0
                self._obstacle_missing_frames = 0
                correction = self._obstacle_direction * max(abs(raw_correction), avoid_min_corr)
                severity = max(raw_severity, avoid_min_sev)
                return correction, severity

            progress = self._clip(self._obstacle_recover_countdown / float(recover_frames), 0.0, 1.0)
            correction = -self._obstacle_direction * recover_max_corr * progress
            severity = recover_min_sev * progress

            self._obstacle_recover_countdown -= 1
            if self._obstacle_recover_countdown <= 0:
                self._obstacle_phase = 'idle'
                self._obstacle_direction = 0.0
                self._obstacle_missing_frames = 0
                correction = 0.0
                severity = 0.0

            return correction, severity

        return raw_correction, raw_severity

    def run(
        self,
        rgb_img,
        pilot_angle=0.0,
        yolo_detector_ready=None,
        tf_detector_ready=None,
        active_profile=None,
    ) -> Tuple[float, float, float, float]:
        pilot_angle = self._clip(float(pilot_angle), -1.0, 1.0)
        if rgb_img is None:
            return pilot_angle, 0.0, 0.0, 0.0

        lane_angle, lane_confidence = self._lane_signal(rgb_img, pilot_angle)
        obstacle_correction, obstacle_severity = self._obstacle_signal(rgb_img)
        obstacle_correction, obstacle_severity = self._obstacle_lane_change_policy(
            obstacle_correction,
            obstacle_severity,
            active_profile=active_profile,
        )

        detector_down = self._detector_down(yolo_detector_ready, tf_detector_ready)
        if detector_down and self.reduce_signals_when_detector_down:
            scale = self._clip(self.detector_down_signal_scale, 0.0, 1.0)
            lane_confidence *= scale
            obstacle_severity *= scale
            obstacle_correction *= scale

        lane_angle = self._clip(lane_angle, -self.lane_angle_max_abs, self.lane_angle_max_abs)
        lane_confidence = self._clip(lane_confidence, 0.0, 1.0)
        obstacle_correction = self._clip(obstacle_correction, -self.obstacle_max_correction, self.obstacle_max_correction)
        obstacle_severity = self._clip(obstacle_severity, 0.0, 1.0)

        if self.enable_temporal_smoothing:
            lane_angle = self._ema(self._lane_angle_ema, lane_angle)
            lane_confidence = self._ema(self._lane_confidence_ema, lane_confidence)
            obstacle_correction = self._ema(self._obstacle_correction_ema, obstacle_correction)
            obstacle_severity = self._ema(self._obstacle_severity_ema, obstacle_severity)

            self._lane_angle_ema = lane_angle
            self._lane_confidence_ema = lane_confidence
            self._obstacle_correction_ema = obstacle_correction
            self._obstacle_severity_ema = obstacle_severity

            lane_angle = self._clip(lane_angle, -self.lane_angle_max_abs, self.lane_angle_max_abs)
            lane_confidence = self._clip(lane_confidence, 0.0, 1.0)
            obstacle_correction = self._clip(obstacle_correction, -self.obstacle_max_correction, self.obstacle_max_correction)
            obstacle_severity = self._clip(obstacle_severity, 0.0, 1.0)

        if self.debug:
            print(
                "[FIRA SIGNALS] "
                f"lane_angle={lane_angle:.3f} lane_conf={lane_confidence:.3f} "
                f"obs_corr={obstacle_correction:.3f} obs_sev={obstacle_severity:.3f} "
                f"obs_phase={self._obstacle_phase} profile={str(active_profile or 'CFG').upper()}"
            )

        return lane_angle, lane_confidence, obstacle_correction, obstacle_severity