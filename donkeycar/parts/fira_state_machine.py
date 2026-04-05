import time
from typing import Optional, Tuple


class FiraDriveStateMachine:
    """Map FIRA engine context to a small set of competition drive states."""

    STOP_STATES = {'stop'}
    ZEBRA_STATES = {'wait-for-crosswalk', 'wait-at-crosswalk'}
    URBAN_STATES = {'pre_turn_forward', 'turn_left', 'turn_right', 'after_turn_sequence', 'proceeding'}

    def __init__(
        self,
        startup_hold_seconds: float = 1.0,
        recovery_hold_seconds: float = 0.8,
        stop_hold_seconds: float = 0.5,
        zebra_hold_seconds: float = 0.35,
        lane_confidence_min: float = 0.25,
        obstacle_severity_min: float = 0.75,
        race_fast_throttle_cap: float = 1.0,
        urban_throttle_cap: float = 0.65,
        zebra_throttle_cap: float = 0.25,
        stop_throttle_cap: float = 0.0,
        recovery_throttle_cap: float = 0.15,
        detector_down_throttle_cap: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.startup_hold_seconds = float(startup_hold_seconds)
        self.recovery_hold_seconds = float(recovery_hold_seconds)
        self.stop_hold_seconds = float(stop_hold_seconds)
        self.zebra_hold_seconds = float(zebra_hold_seconds)
        self.lane_confidence_min = float(lane_confidence_min)
        self.obstacle_severity_min = float(obstacle_severity_min)
        self.race_fast_throttle_cap = float(race_fast_throttle_cap)
        self.urban_throttle_cap = float(urban_throttle_cap)
        self.zebra_throttle_cap = float(zebra_throttle_cap)
        self.stop_throttle_cap = float(stop_throttle_cap)
        self.recovery_throttle_cap = float(recovery_throttle_cap)
        self.detector_down_throttle_cap = float(detector_down_throttle_cap)
        self.debug = bool(debug)

        self.state = 'startup'
        self._state_enter_time: Optional[float] = None
        self._last_reason = 'boot'

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _normalize_state(state) -> str:
        if state is None:
            return ''
        return str(state).strip().lower().replace(' ', '_')

    @staticmethod
    def _health_is_ready(health) -> bool:
        if not isinstance(health, dict):
            return True
        if health.get('detector_backend', 'available') == 'unavailable':
            return False
        return bool(health.get('detector_ready', True))

    def _state_age(self, now: float) -> float:
        if self._state_enter_time is None:
            return 0.0
        return max(0.0, now - self._state_enter_time)

    def _set_state(self, new_state: str, now: float, reason: str) -> None:
        if self.state != new_state:
            self.state = new_state
            self._state_enter_time = now
        elif self._state_enter_time is None:
            self._state_enter_time = now
        self._last_reason = reason

    def _hold_state(self, now: float) -> bool:
        age = self._state_age(now)
        if self.state == 'startup':
            return age < self.startup_hold_seconds
        if self.state == 'recovery':
            return age < self.recovery_hold_seconds
        if self.state == 'stop_hold':
            return age < self.stop_hold_seconds
        if self.state == 'zebra_slow':
            return age < self.zebra_hold_seconds
        return False

    def _state_cap(self, state: str) -> float:
        if state == 'startup':
            return 0.0
        if state == 'recovery':
            if self._last_reason == 'detector_down':
                return self._clip(self.detector_down_throttle_cap, 0.0, 1.0)
            return self._clip(self.recovery_throttle_cap, 0.0, 1.0)
        if state == 'stop_hold':
            return self._clip(self.stop_throttle_cap, 0.0, 1.0)
        if state == 'zebra_slow':
            return self._clip(self.zebra_throttle_cap, 0.0, 1.0)
        if state == 'urban_navigate':
            return self._clip(self.urban_throttle_cap, 0.0, 1.0)
        return self._clip(self.race_fast_throttle_cap, 0.0, 1.0)

    def _first_state(self, *states: Optional[object]) -> str:
        for state in states:
            normalized = self._normalize_state(state)
            if normalized:
                return normalized
        return ''

    def run(
        self,
        yolo_state=None,
        tf_state=None,
        yolo_health=None,
        tf_health=None,
        lane_confidence=None,
        obstacle_severity=None,
    ) -> Tuple[str, float]:
        now = time.time()

        if self._state_enter_time is None:
            self._state_enter_time = now

        if self._hold_state(now):
            if self.debug:
                print(f"[FIRA STATE] hold={self.state} reason={self._last_reason}")
            return self.state, self._state_cap(self.state)

        detector_down = not self._health_is_ready(yolo_health) or not self._health_is_ready(tf_health)
        lane_conf = None if lane_confidence is None else self._clip(float(lane_confidence), 0.0, 1.0)
        obstacle_level = None if obstacle_severity is None else self._clip(float(obstacle_severity), 0.0, 1.0)
        engine_state = self._first_state(yolo_state, tf_state)

        if detector_down:
            self._set_state('recovery', now, 'detector_down')
        elif engine_state in self.STOP_STATES:
            self._set_state('stop_hold', now, f'engine_state={engine_state}')
        elif engine_state in self.ZEBRA_STATES:
            self._set_state('zebra_slow', now, f'engine_state={engine_state}')
        elif engine_state in self.URBAN_STATES:
            self._set_state('urban_navigate', now, f'engine_state={engine_state}')
        elif lane_conf is not None and lane_conf < self.lane_confidence_min:
            self._set_state('recovery', now, f'low_lane_conf={lane_conf:.3f}')
        elif obstacle_level is not None and obstacle_level >= self.obstacle_severity_min:
            self._set_state('zebra_slow', now, f'high_obstacle={obstacle_level:.3f}')
        else:
            self._set_state('race_fast', now, 'nominal')

        cap = self._state_cap(self.state)

        if self.debug:
            print(
                f"[FIRA STATE] state={self.state} cap={cap:.2f} "
                f"reason={self._last_reason} lane={lane_conf} obstacle={obstacle_level}"
            )

        return self.state, cap
