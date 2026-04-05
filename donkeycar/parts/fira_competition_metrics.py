from __future__ import annotations

from typing import Optional, Tuple


class FiraCompetitionMetrics:
    """Track lightweight competition compliance metrics for Urban/Race runs.

    The tracker is intentionally conservative and map-agnostic:
    - right_lane_score: fraction of active frames considered right-lane compliant
        - checkpoint_count: real event-driven checkpoints when available; otherwise
            fallback proxy increments on stable right-lane streaks
    - checkpoint_progress: normalized progress against configured target checkpoints
    - lane_violations: frames in active driving with low lane confidence
    - active_frames: number of frames considered in scoring window
    """

    ACTIVE_STATES = {'urban_navigate', 'race_fast', 'conduciendo', 'proceeding'}

    def __init__(
        self,
        enabled: bool = True,
        lane_conf_min: float = 0.45,
        right_lane_angle_min: float = -0.28,
        right_lane_angle_max: float = 0.08,
        checkpoint_min_right_lane_streak: int = 20,
        checkpoint_cooldown_frames: int = 32,
        target_checkpoints: int = 12,
        compliance_score_threshold: float = 0.72,
    ) -> None:
        self.enabled = bool(enabled)
        self.lane_conf_min = float(lane_conf_min)
        self.right_lane_angle_min = float(right_lane_angle_min)
        self.right_lane_angle_max = float(right_lane_angle_max)
        self.checkpoint_min_right_lane_streak = max(1, int(checkpoint_min_right_lane_streak))
        self.checkpoint_cooldown_frames = max(0, int(checkpoint_cooldown_frames))
        self.target_checkpoints = max(1, int(target_checkpoints))
        self.compliance_score_threshold = float(compliance_score_threshold)

        self.active_frames = 0
        self.right_lane_frames = 0
        self.lane_violations = 0
        self.checkpoint_count = 0

        self.compliance_score = 0.0
        self.compliance_ready = False

        self._right_lane_streak = 0
        self._checkpoint_cooldown = 0
        self._last_checkpoint_event = False
        self._uses_real_checkpoint_events = False

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_state(value) -> str:
        if value is None:
            return ''
        return str(value).strip().lower()

    def _is_active_state(self, drive_state: str) -> bool:
        if not drive_state:
            return False
        if drive_state in self.ACTIVE_STATES:
            return True
        # Keep compatibility with free-form states such as "turn_right".
        return ('turn' in drive_state) or ('drive' in drive_state)

    def _compute_compliance_score(self, right_lane_score: float, checkpoint_progress: float) -> float:
        violation_rate = 0.0
        if self.active_frames > 0:
            violation_rate = float(self.lane_violations) / float(self.active_frames)

        # Weighted score: right-lane discipline + checkpoint completion - lane violations.
        score = (0.55 * right_lane_score) + (0.45 * checkpoint_progress) - (0.20 * violation_rate)
        return self._clip(score, 0.0, 1.0)

    def get_report(self) -> dict:
        right_lane_score = 0.0
        if self.active_frames > 0:
            right_lane_score = float(self.right_lane_frames) / float(self.active_frames)
        return {
            'right_lane_score': float(right_lane_score),
            'checkpoint_count': int(self.checkpoint_count),
            'checkpoint_progress': self._clip(float(self.checkpoint_count) / float(self.target_checkpoints), 0.0, 1.0),
            'checkpoint_source': 'event' if self._uses_real_checkpoint_events else 'proxy',
            'lane_violations': int(self.lane_violations),
            'active_frames': int(self.active_frames),
            'compliance_score': float(self.compliance_score),
            'compliance_ready': bool(self.compliance_ready),
        }

    def run(
        self,
        lane_confidence=None,
        lane_angle=None,
        drive_state=None,
        obstacle_severity=None,
        checkpoint_event=None,
    ) -> Tuple[float, int, float, int, int, float, bool]:
        if not self.enabled:
            return 0.0, 0, 0.0, 0, 0, 0.0, False

        lane_conf = self._clip(self._as_float(lane_confidence, 0.0), 0.0, 1.0)
        lane_ang = self._as_float(lane_angle, 0.0)
        state = self._normalize_state(drive_state)
        _ = self._clip(self._as_float(obstacle_severity, 0.0), 0.0, 1.0)
        ckpt_event = self._as_float(checkpoint_event, 0.0) >= 0.5

        active = self._is_active_state(state)
        right_lane_ok = (
            active and
            lane_conf >= self.lane_conf_min and
            self.right_lane_angle_min <= lane_ang <= self.right_lane_angle_max
        )

        if active:
            self.active_frames += 1
            if lane_conf < self.lane_conf_min:
                self.lane_violations += 1

        if right_lane_ok:
            self.right_lane_frames += 1
            self._right_lane_streak += 1
        else:
            self._right_lane_streak = 0

        if self._checkpoint_cooldown > 0:
            self._checkpoint_cooldown -= 1

        if ckpt_event and (not self._last_checkpoint_event):
            # Rising-edge event: count physically detected checkpoint crossing.
            self._uses_real_checkpoint_events = True
            self.checkpoint_count += 1
            if not right_lane_ok and active:
                # Explicitly penalize checkpoint crossing while not right-lane compliant.
                self.lane_violations += 1
            self._checkpoint_cooldown = self.checkpoint_cooldown_frames
            self._right_lane_streak = 0
        elif (
            not self._uses_real_checkpoint_events and
            self._right_lane_streak >= self.checkpoint_min_right_lane_streak and
            self._checkpoint_cooldown == 0
        ):
            self.checkpoint_count += 1
            self._checkpoint_cooldown = self.checkpoint_cooldown_frames
            self._right_lane_streak = 0

        self._last_checkpoint_event = ckpt_event

        right_lane_score = 0.0
        if self.active_frames > 0:
            right_lane_score = float(self.right_lane_frames) / float(self.active_frames)

        checkpoint_progress = self._clip(
            float(self.checkpoint_count) / float(self.target_checkpoints),
            0.0,
            1.0,
        )

        self.compliance_score = self._compute_compliance_score(right_lane_score, checkpoint_progress)
        self.compliance_ready = bool(self.compliance_score >= self.compliance_score_threshold)

        return (
            right_lane_score,
            int(self.checkpoint_count),
            checkpoint_progress,
            int(self.lane_violations),
            int(self.active_frames),
            float(self.compliance_score),
            bool(self.compliance_ready),
        )
