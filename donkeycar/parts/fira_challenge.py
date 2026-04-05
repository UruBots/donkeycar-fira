from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Detection:
    cls: str
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def cx(self) -> float:
        return float(self.x + 0.5 * self.w)

    @property
    def cy(self) -> float:
        return float(self.y + 0.5 * self.h)

    @property
    def right(self) -> float:
        return float(self.x + self.w)

    @property
    def bottom(self) -> float:
        return float(self.y + self.h)


class FiraChallengePart:
    """Evasive race policy for moving cars/cones while staying inside lane bounds.

    The part consumes pilot angle/throttle and camera image, detects obstacles using
    HSV + contours, and applies a bounded correction. It prioritizes the nearest
    obstacle (front-most) and supports sequential obstacles with temporal smoothing.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_abs_steering: float = 1.0,
        max_abs_correction: float = 0.7,
        steering_gain: float = 0.9,
        max_steering_delta_per_sec: float = 2.0,
        min_throttle_factor: float = 0.45,
        lane_color_low: Tuple[int, int, int] = (18, 70, 70),
        lane_color_high: Tuple[int, int, int] = (42, 255, 255),
        lane_roi_top_ratio: float = 0.45,
        lane_min_pixel_ratio: float = 0.002,
        lane_margin_px: int = 10,
        car_color_low: Tuple[int, int, int] = (0, 0, 25),
        car_color_high: Tuple[int, int, int] = (179, 95, 185),
        cone_color_low: Tuple[int, int, int] = (5, 110, 80),
        cone_color_high: Tuple[int, int, int] = (25, 255, 255),
        obstacle_roi_top_ratio: float = 0.35,
        obstacle_roi_bottom_ratio: float = 0.98,
        obstacle_min_area_ratio: float = 0.002,
        obstacle_full_area_ratio: float = 0.06,
        obstacle_max_area_ratio: float = 0.45,
        obstacle_min_aspect: float = 0.25,
        obstacle_max_aspect: float = 4.0,
        nearest_weight: float = 0.7,
        clearance_px: int = 18,
        engage_severity: float = 0.12,
        clear_frames: int = 2,
        recover_frames: int = 6,
        smoothing_alpha: float = 0.35,
        warmup_frames: int = 0,
        warmup_steering_gain: float = 0.18,
        warmup_max_abs_angle: float = 0.25,
        warmup_throttle: Optional[float] = None,
        predictive_enabled: bool = False,
        predictive_horizon_sec: float = 0.20,
        predictive_blend: float = 0.55,
        predictive_velocity_alpha: float = 0.45,
        predictive_min_confidence: float = 0.25,
        predictive_max_missing_frames: int = 3,
        collision_imminent_enabled: bool = True,
        collision_imminent_proximity: float = 0.88,
        collision_imminent_occupancy: float = 0.42,
        collision_imminent_throttle_cap: float = 0.02,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_abs_steering = float(max_abs_steering)
        self.max_abs_correction = float(max_abs_correction)
        self.steering_gain = float(steering_gain)
        self.max_steering_delta_per_sec = float(max_steering_delta_per_sec)
        self.min_throttle_factor = float(min_throttle_factor)

        self.lane_color_low = np.array(lane_color_low, dtype=np.uint8)
        self.lane_color_high = np.array(lane_color_high, dtype=np.uint8)
        self.lane_roi_top_ratio = float(lane_roi_top_ratio)
        self.lane_min_pixel_ratio = float(lane_min_pixel_ratio)
        self.lane_margin_px = int(lane_margin_px)

        self.car_color_low = np.array(car_color_low, dtype=np.uint8)
        self.car_color_high = np.array(car_color_high, dtype=np.uint8)
        self.cone_color_low = np.array(cone_color_low, dtype=np.uint8)
        self.cone_color_high = np.array(cone_color_high, dtype=np.uint8)
        self.obstacle_roi_top_ratio = float(obstacle_roi_top_ratio)
        self.obstacle_roi_bottom_ratio = float(obstacle_roi_bottom_ratio)
        self.obstacle_min_area_ratio = float(obstacle_min_area_ratio)
        self.obstacle_full_area_ratio = float(obstacle_full_area_ratio)
        self.obstacle_max_area_ratio = float(obstacle_max_area_ratio)
        self.obstacle_min_aspect = float(obstacle_min_aspect)
        self.obstacle_max_aspect = float(obstacle_max_aspect)
        self.nearest_weight = float(nearest_weight)
        self.clearance_px = int(clearance_px)

        self.engage_severity = float(engage_severity)
        self.clear_frames = max(1, int(clear_frames))
        self.recover_frames = max(1, int(recover_frames))
        self.smoothing_alpha = float(smoothing_alpha)
        self.warmup_frames = max(0, int(warmup_frames))
        self.warmup_steering_gain = float(warmup_steering_gain)
        self.warmup_max_abs_angle = float(warmup_max_abs_angle)
        self.warmup_throttle = None if warmup_throttle is None else float(warmup_throttle)
        self.predictive_enabled = bool(predictive_enabled)
        self.predictive_horizon_sec = float(predictive_horizon_sec)
        self.predictive_blend = float(predictive_blend)
        self.predictive_velocity_alpha = float(predictive_velocity_alpha)
        self.predictive_min_confidence = float(predictive_min_confidence)
        self.predictive_max_missing_frames = max(1, int(predictive_max_missing_frames))
        self.collision_imminent_enabled = bool(collision_imminent_enabled)
        self.collision_imminent_proximity = float(collision_imminent_proximity)
        self.collision_imminent_occupancy = float(collision_imminent_occupancy)
        self.collision_imminent_throttle_cap = float(collision_imminent_throttle_cap)
        self.debug = bool(debug)

        self._state = 'idle'
        self._missing_frames = 0
        self._recover_countdown = 0
        self._last_correction = 0.0
        self._last_timestamp: Optional[float] = None
        self._warmup_step = 0
        self._track_prev_cx: Optional[float] = None
        self._track_prev_time: Optional[float] = None
        self._track_velocity_x = 0.0
        self._track_confidence = 0.0
        self._track_missing_frames = 0

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _ema(self, prev: float, now: float) -> float:
        alpha = self._clip(self.smoothing_alpha, 0.0, 1.0)
        return (1.0 - alpha) * prev + alpha * now

    def _lane_corridor(self, rgb_img: np.ndarray) -> Tuple[float, float, float]:
        """Estimate drivable horizontal corridor [left, right] and confidence."""
        h, w = rgb_img.shape[:2]
        top = int(self._clip(self.lane_roi_top_ratio, 0.0, 0.98) * h)
        roi = rgb_img[top:, :, :]
        if roi.size == 0:
            return float(self.lane_margin_px), float(w - self.lane_margin_px), 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lane_color_low, self.lane_color_high)

        active = np.where(mask > 0)
        ratio = float(active[0].size) / float(mask.size) if mask.size > 0 else 0.0
        if active[1].size < 20:
            return float(self.lane_margin_px), float(w - self.lane_margin_px), 0.0

        x_vals = active[1]
        left = float(np.percentile(x_vals, 5.0))
        right = float(np.percentile(x_vals, 95.0))
        left = self._clip(left + self.lane_margin_px, 0.0, float(w - 2))
        right = self._clip(right - self.lane_margin_px, left + 2.0, float(w - 1))

        conf = self._clip(ratio / max(self.lane_min_pixel_ratio, 1e-6), 0.0, 1.0)
        return left, right, conf

    def _extract(self, hsv_roi: np.ndarray, cls_name: str, low: np.ndarray, high: np.ndarray) -> List[_Detection]:
        mask = cv2.inRange(hsv_roi, low, high)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[_Detection] = []
        total_area = float(mask.shape[0] * mask.shape[1])
        for c in contours:
            area = float(cv2.contourArea(c))
            if total_area <= 0.0:
                continue
            area_ratio = area / total_area
            if area_ratio < self.obstacle_min_area_ratio:
                continue
            if area_ratio > self.obstacle_max_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(c)
            roi_h, roi_w = mask.shape[:2]
            # Ignore large/background contours touching ROI boundaries.
            if x <= 1 or y <= 1 or (x + w) >= (roi_w - 1) or (y + h) >= (roi_h - 1):
                continue
            aspect = float(w) / float(max(1, h))
            if aspect < self.obstacle_min_aspect or aspect > self.obstacle_max_aspect:
                continue
            span = max(1e-6, self.obstacle_full_area_ratio - self.obstacle_min_area_ratio)
            score = self._clip((area_ratio - self.obstacle_min_area_ratio) / span, 0.0, 1.0)
            dets.append(_Detection(cls=cls_name, x=x, y=y, w=w, h=h, score=score))
        return dets

    def _detect_obstacles(self, rgb_img: np.ndarray) -> List[_Detection]:
        h, w = rgb_img.shape[:2]
        top = int(self._clip(self.obstacle_roi_top_ratio, 0.0, 0.98) * h)
        bottom = int(self._clip(self.obstacle_roi_bottom_ratio, 0.02, 1.0) * h)
        if bottom <= top:
            return []

        roi = rgb_img[top:bottom, :, :]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        dets = self._extract(hsv, 'car', self.car_color_low, self.car_color_high)
        dets.extend(self._extract(hsv, 'cone', self.cone_color_low, self.cone_color_high))

        # Re-map y from roi to full image coordinates.
        for d in dets:
            d.y += top
        return dets

    def _pick_primary(self, dets: List[_Detection], image_h: int) -> Optional[_Detection]:
        if not dets:
            return None

        # Nearest-first with score tie-break: bottom-most obstacle is considered closest.
        def rank(d: _Detection) -> float:
            near = self._clip(d.bottom / max(1.0, float(image_h)), 0.0, 1.0)
            return self.nearest_weight * near + (1.0 - self.nearest_weight) * d.score

        return max(dets, key=rank)

    def _desired_target_x(
        self,
        obstacle: _Detection,
        lane_left: float,
        lane_right: float,
        lane_center: float,
    ) -> float:
        gap_left = max(0.0, obstacle.x - lane_left)
        gap_right = max(0.0, lane_right - obstacle.right)

        # Obstacle left -> go right, obstacle right -> go left. If chosen side has
        # insufficient space, choose the opposite side.
        prefer_right = obstacle.cx < lane_center
        min_gap = float(self.clearance_px + 0.5 * obstacle.w)
        if prefer_right and gap_right < min_gap and gap_left > gap_right:
            prefer_right = False
        elif (not prefer_right) and gap_left < min_gap and gap_right > gap_left:
            prefer_right = True

        if prefer_right:
            target = lane_center + 0.55 * gap_right
        else:
            target = lane_center - 0.55 * gap_left
        return self._clip(target, lane_left + 2.0, lane_right - 2.0)

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        merged: List[Tuple[float, float]] = []
        for start, end in sorted(intervals, key=lambda v: (v[0], v[1])):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _gap_target_x(
        self,
        dets: List[_Detection],
        lane_left: float,
        lane_right: float,
        lane_center: float,
        image_h: int,
    ) -> float:
        lane_width = max(1.0, lane_right - lane_left)

        blocked: List[Tuple[float, float]] = []
        for d in dets:
            start = self._clip(d.x - self.clearance_px, lane_left, lane_right)
            end = self._clip(d.right + self.clearance_px, lane_left, lane_right)
            if end - start >= 1.0:
                blocked.append((start, end))

        merged = self._merge_intervals(blocked)
        gaps: List[Tuple[float, float]] = []
        cursor = lane_left
        for start, end in merged:
            if start - cursor >= 2.0:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if lane_right - cursor >= 2.0:
            gaps.append((cursor, lane_right))

        if not gaps:
            # If all corridor is blocked, bias to lane center and let
            # correction saturation/velocity limits keep behavior stable.
            return lane_center

        max_proximity = max(self._clip(d.bottom / max(1.0, float(image_h)), 0.0, 1.0) for d in dets)
        best_score = -1e9
        best_center = lane_center
        for g_left, g_right in gaps:
            center = 0.5 * (g_left + g_right)
            width_norm = self._clip((g_right - g_left) / lane_width, 0.0, 1.0)
            center_align = 1.0 - self._clip(abs(center - lane_center) / (0.5 * lane_width), 0.0, 1.0)

            risk = 0.0
            for d in dets:
                proximity = self._clip(d.bottom / max(1.0, float(image_h)), 0.0, 1.0)
                spread = max(6.0, 0.5 * d.w + float(self.clearance_px))
                influence = np.exp(-abs(center - d.cx) / spread)
                risk += proximity * influence
            risk = self._clip(risk / max(1.0, float(len(dets))), 0.0, 1.0)

            # Near obstacles should prioritize low-risk gaps more strongly.
            risk_weight = 0.50 + 0.35 * max_proximity
            score = 0.55 * width_norm + 0.30 * center_align - risk_weight * risk
            if score > best_score:
                best_score = score
                best_center = center

        return self._clip(best_center, lane_left + 2.0, lane_right - 2.0)

    def _correction_from_target(self, target_x: float, lane_left: float, lane_right: float) -> float:
        lane_center = 0.5 * (lane_left + lane_right)
        half = max(6.0, 0.5 * (lane_right - lane_left))
        err = (target_x - lane_center) / half
        return self._clip(err * self.steering_gain, -self.max_abs_correction, self.max_abs_correction)

    def _rate_limit(self, value: float, now: float) -> float:
        if self._last_timestamp is None:
            self._last_timestamp = now
            return value
        dt = max(1e-3, now - self._last_timestamp)
        self._last_timestamp = now
        max_delta = self.max_steering_delta_per_sec * dt
        delta = self._clip(value - self._last_correction, -max_delta, max_delta)
        return self._last_correction + delta

    def _update_track(self, primary: Optional[_Detection], now: float) -> Tuple[Optional[_Detection], float]:
        if not self.predictive_enabled:
            return None, 0.0

        if primary is None:
            self._track_missing_frames += 1
            self._track_confidence *= 0.6
            if self._track_missing_frames > self.predictive_max_missing_frames:
                self._track_prev_cx = None
                self._track_prev_time = None
                self._track_velocity_x = 0.0
                self._track_confidence = 0.0
            return None, self._track_confidence

        self._track_missing_frames = 0
        curr_cx = primary.cx
        if self._track_prev_cx is not None and self._track_prev_time is not None:
            dt = max(1e-3, now - self._track_prev_time)
            measured_vx = (curr_cx - self._track_prev_cx) / dt
            vel_alpha = self._clip(self.predictive_velocity_alpha, 0.0, 1.0)
            self._track_velocity_x = (1.0 - vel_alpha) * self._track_velocity_x + vel_alpha * measured_vx

            disp = abs(curr_cx - self._track_prev_cx)
            scale = max(6.0, 1.8 * float(primary.w))
            meas_conf = 1.0 - self._clip(disp / scale, 0.0, 1.0)
            self._track_confidence = self._clip(0.45 * self._track_confidence + 0.55 * meas_conf, 0.0, 1.0)
        else:
            self._track_velocity_x = 0.0
            self._track_confidence = max(self._track_confidence, 0.40)

        self._track_prev_cx = curr_cx
        self._track_prev_time = now

        horizon = max(0.0, self.predictive_horizon_sec)
        pred_cx = curr_cx + self._track_velocity_x * horizon
        pred_x = int(round(pred_cx - 0.5 * primary.w))
        predicted = _Detection(
            cls=primary.cls,
            x=pred_x,
            y=primary.y,
            w=primary.w,
            h=primary.h,
            score=primary.score,
        )
        return predicted, self._track_confidence

    def run(
        self,
        pilot_angle: float,
        pilot_throttle: float,
        rgb_img,
        lane_confidence: Optional[float] = None,
        obstacle_severity_hint: Optional[float] = None,
    ) -> Tuple[float, float, object]:
        angle = self._clip(float(pilot_angle), -self.max_abs_steering, self.max_abs_steering)
        throttle = float(pilot_throttle)
        if not self.enabled or rgb_img is None:
            return angle, throttle, rgb_img

        h, w = rgb_img.shape[:2]
        lane_left, lane_right, local_lane_conf = self._lane_corridor(rgb_img)
        lane_center = 0.5 * (lane_left + lane_right)
        lane_conf = local_lane_conf if lane_confidence is None else self._clip(float(lane_confidence), 0.0, 1.0)

        dets = self._detect_obstacles(rgb_img)
        primary = self._pick_primary(dets, h)
        now_ts = time.monotonic()
        predicted_primary, track_conf = self._update_track(primary, now_ts)

        raw_correction = 0.0
        severity = 0.0
        imminent_collision = False
        if primary is not None:
            if len(dets) == 1:
                target_x = self._desired_target_x(primary, lane_left, lane_right, lane_center)
            else:
                target_x = self._gap_target_x(dets, lane_left, lane_right, lane_center, h)
            raw_correction = self._correction_from_target(target_x, lane_left, lane_right)

            if (
                predicted_primary is not None
                and track_conf >= self._clip(self.predictive_min_confidence, 0.0, 1.0)
            ):
                pred_target_x = self._desired_target_x(predicted_primary, lane_left, lane_right, lane_center)
                pred_correction = self._correction_from_target(pred_target_x, lane_left, lane_right)
                blend = self._clip(self.predictive_blend, 0.0, 1.0) * self._clip(track_conf, 0.0, 1.0)
                raw_correction = (1.0 - blend) * raw_correction + blend * pred_correction

            proximity = self._clip(primary.bottom / max(1.0, float(h)), 0.0, 1.0)
            occupancy = self._clip(primary.w / max(1.0, (lane_right - lane_left)), 0.0, 1.0)
            severity = self._clip(0.65 * proximity + 0.35 * occupancy, 0.0, 1.0)
            imminent_collision = (
                self.collision_imminent_enabled
                and proximity >= self.collision_imminent_proximity
                and occupancy >= self.collision_imminent_occupancy
            )

        if obstacle_severity_hint is not None:
            severity = max(severity, self._clip(float(obstacle_severity_hint), 0.0, 1.0))

        # State machine for sequential obstacles and stable S-maneuvers.
        obstacle_seen = severity >= self.engage_severity
        transitioned_to_avoid = False
        if self._state == 'idle':
            if obstacle_seen:
                self._state = 'avoid'
                self._missing_frames = 0
            transitioned_to_avoid = True

        if self._state == 'avoid':
            if obstacle_seen:
                self._missing_frames = 0
            else:
                self._missing_frames += 1
                if self._missing_frames >= self.clear_frames:
                    self._state = 'recover'
                    self._recover_countdown = self.recover_frames

        if self._state == 'recover':
            if obstacle_seen:
                self._state = 'avoid'
                self._missing_frames = 0
                transitioned_to_avoid = True
            else:
                progress = self._clip(self._recover_countdown / float(self.recover_frames), 0.0, 1.0)
                raw_correction = self._last_correction * progress
                severity *= progress
                self._recover_countdown -= 1
                if self._recover_countdown <= 0:
                    self._state = 'idle'
                    raw_correction = 0.0
                    severity = 0.0

        if self._state == 'idle' and not obstacle_seen:
            raw_correction = 0.0
            severity = 0.0

        # Startup warmup: use gentle lane recentering before enabling full obstacle authority.
        warmup_active = self._warmup_step < self.warmup_frames
        if warmup_active:
            img_center = 0.5 * (float(w) - 1.0)
            half = max(6.0, 0.5 * (lane_right - lane_left))
            lane_err = (lane_center - img_center) / half
            warmup_corr = self._clip(
                lane_err * self.warmup_steering_gain,
                -self.warmup_max_abs_angle,
                self.warmup_max_abs_angle,
            )
            ramp = self._clip((self._warmup_step + 1) / max(1.0, float(self.warmup_frames)), 0.0, 1.0)
            raw_correction = (1.0 - ramp) * warmup_corr + ramp * raw_correction
            severity *= ramp

        # If lane confidence is low, become conservative.
        lane_scale = 0.35 + 0.65 * lane_conf
        raw_correction *= lane_scale
        severity *= lane_scale

        # For nearest-first S maneuvers, re-planning must react fast when a new
        # obstacle appears during recovery/transition.
        if transitioned_to_avoid:
            self._last_timestamp = None

        smooth = self._ema(self._last_correction, raw_correction)
        limited = self._rate_limit(smooth, now_ts)
        self._last_correction = self._clip(limited, -self.max_abs_correction, self.max_abs_correction)

        out_angle = self._clip(angle + self._last_correction, -self.max_abs_steering, self.max_abs_steering)

        sev = self._clip(severity, 0.0, 1.0)
        factor = 1.0 - sev * (1.0 - self._clip(self.min_throttle_factor, 0.0, 1.0))
        out_throttle = throttle * factor
        if warmup_active and self.warmup_throttle is not None:
            out_throttle = min(out_throttle, self.warmup_throttle)
        if imminent_collision:
            out_throttle = min(out_throttle, self.collision_imminent_throttle_cap)

        if warmup_active:
            self._warmup_step += 1

        if self.debug:
            print(
                f"[FiraChallenge] state={self._state} dets={len(dets)} "
                f"corr={self._last_correction:.3f} sev={sev:.3f} "
                f"lane_conf={lane_conf:.2f} warmup={int(warmup_active)} "
                f"track_conf={track_conf:.2f}"
            )

        return out_angle, out_throttle, rgb_img
