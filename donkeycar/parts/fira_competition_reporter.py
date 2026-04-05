from __future__ import annotations

import json
import os
import time
from typing import Optional


class FiraCompetitionSessionReporter:
    """Write session compliance report when recording stops.

    The reporter listens to recording transitions:
    - recording False -> True: starts a session window
    - recording True -> False: writes a JSON report if enough active frames exist
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        min_active_frames: int = 120,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.min_active_frames = max(1, int(min_active_frames))

        self._last_recording: Optional[bool] = None
        self._session_started_at: Optional[float] = None

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _write_report(self, payload: dict) -> None:
        self._ensure_output_dir()
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.output_dir, f'fira_competition_report_{ts}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def run(
        self,
        recording=None,
        user_mode=None,
        right_lane_score=None,
        checkpoint_count=None,
        checkpoint_progress=None,
        lane_violations=None,
        active_frames=None,
        compliance_score=None,
        compliance_ready=None,
    ):
        if not self.enabled:
            return

        is_recording = bool(recording)

        if self._last_recording is None:
            self._last_recording = is_recording
            if is_recording:
                self._session_started_at = time.time()
            return

        # Start session window.
        if (not self._last_recording) and is_recording:
            self._session_started_at = time.time()

        # Stop session window: emit report.
        if self._last_recording and (not is_recording):
            active = self._as_int(active_frames, 0)
            if active >= self.min_active_frames:
                now = time.time()
                payload = {
                    'generated_at_epoch': now,
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
                    'session_started_at_epoch': self._session_started_at,
                    'session_duration_sec': (now - self._session_started_at) if self._session_started_at else None,
                    'user_mode': '' if user_mode is None else str(user_mode),
                    'metrics': {
                        'right_lane_score': self._as_float(right_lane_score, 0.0),
                        'checkpoint_count': self._as_int(checkpoint_count, 0),
                        'checkpoint_progress': self._as_float(checkpoint_progress, 0.0),
                        'lane_violations': self._as_int(lane_violations, 0),
                        'active_frames': active,
                        'compliance_score': self._as_float(compliance_score, 0.0),
                        'compliance_ready': bool(compliance_ready),
                    },
                }
                self._write_report(payload)

        self._last_recording = is_recording
