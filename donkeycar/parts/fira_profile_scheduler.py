from typing import Optional


class FiraProfileScheduler:
    """Select active obstacle profile at runtime with hysteresis and dwell."""

    VALID_PROFILES = ("SAFE", "RACE", "TIGHT")

    def __init__(
        self,
        enabled: bool = False,
        default_profile: str = "SAFE",
        min_dwell_frames: int = 12,
        low_lane_conf_to_safe: float = 0.25,
        race_enter_severity: float = 0.30,
        race_exit_severity: float = 0.18,
        tight_enter_severity: float = 0.60,
        tight_exit_severity: float = 0.45,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.default_profile = self._normalize_profile(default_profile)
        self.min_dwell_frames = max(0, int(min_dwell_frames))
        self.low_lane_conf_to_safe = float(low_lane_conf_to_safe)
        self.race_enter_severity = float(race_enter_severity)
        self.race_exit_severity = float(race_exit_severity)
        self.tight_enter_severity = float(tight_enter_severity)
        self.tight_exit_severity = float(tight_exit_severity)
        self.debug = bool(debug)

        self._active_profile = self.default_profile
        self._frames_since_switch = 0

    @classmethod
    def _normalize_profile(cls, profile: Optional[str]) -> str:
        if profile is None:
            return "SAFE"
        text = str(profile).strip().upper()
        if text not in cls.VALID_PROFILES:
            return "SAFE"
        return text

    @staticmethod
    def _to_float(value: Optional[object], default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _target_profile(
        self,
        lane_confidence: float,
        obstacle_severity: float,
        drive_state: Optional[str],
    ) -> str:
        state = str(drive_state or "").strip().lower()

        # Non-race contexts should stay conservative.
        if state in {"startup", "stop_hold", "zebra_slow", "urban_navigate", "recovery"}:
            return "SAFE"

        if lane_confidence < self.low_lane_conf_to_safe:
            return "SAFE"

        # Hysteresis around TIGHT profile.
        if self._active_profile == "TIGHT":
            if obstacle_severity >= self.tight_exit_severity:
                return "TIGHT"
        elif obstacle_severity >= self.tight_enter_severity:
            return "TIGHT"

        # Hysteresis around RACE profile.
        if self._active_profile == "RACE":
            if obstacle_severity >= self.race_exit_severity:
                return "RACE"
        elif obstacle_severity >= self.race_enter_severity:
            return "RACE"

        return "SAFE"

    def run(
        self,
        lane_confidence: Optional[float] = None,
        obstacle_severity: Optional[float] = None,
        drive_state: Optional[str] = None,
    ) -> str:
        if not self.enabled:
            return self.default_profile

        lane_conf = self._to_float(lane_confidence, default=1.0)
        obs_sev = self._to_float(obstacle_severity, default=0.0)
        target = self._target_profile(lane_conf, obs_sev, drive_state)

        if target != self._active_profile and self._frames_since_switch >= self.min_dwell_frames:
            self._active_profile = target
            self._frames_since_switch = 0
            if self.debug:
                print(f"[FIRA PROFILE] switched to {self._active_profile}")
        else:
            self._frames_since_switch += 1

        return self._active_profile
