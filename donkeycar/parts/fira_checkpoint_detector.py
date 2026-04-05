from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class FiraCheckpointDetector:
    """Detect visual checkpoint crossing events from camera image.

    The detector is intentionally simple and deterministic: it inspects a lower
    ROI and emits a 1.0 event on rising-edge when checkpoint-color occupancy
    exceeds a threshold. A cooldown avoids duplicate counts while crossing.
    """

    def __init__(
        self,
        enabled: bool = False,
        color_low: Tuple[int, int, int] = (0, 120, 120),
        color_high: Tuple[int, int, int] = (10, 255, 255),
        color_low_2: Tuple[int, int, int] = (170, 120, 120),
        color_high_2: Tuple[int, int, int] = (179, 255, 255),
        roi_top_ratio: float = 0.70,
        min_pixel_ratio: float = 0.08,
        cooldown_frames: int = 12,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.color_low = np.array(color_low, dtype=np.uint8)
        self.color_high = np.array(color_high, dtype=np.uint8)
        self.color_low_2 = np.array(color_low_2, dtype=np.uint8)
        self.color_high_2 = np.array(color_high_2, dtype=np.uint8)
        self.roi_top_ratio = float(roi_top_ratio)
        self.min_pixel_ratio = float(min_pixel_ratio)
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.debug = bool(debug)

        self._cooldown = 0
        self._last_active = False

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def run(self, rgb_img):
        if (not self.enabled) or rgb_img is None:
            return 0.0, 0.0

        h, _ = rgb_img.shape[:2]
        top = int(self._clip(self.roi_top_ratio, 0.0, 0.98) * h)
        roi = rgb_img[top:, :, :]
        if roi.size == 0:
            return 0.0, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask_1 = cv2.inRange(hsv, self.color_low, self.color_high)
        mask_2 = cv2.inRange(hsv, self.color_low_2, self.color_high_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
        active = ratio >= self.min_pixel_ratio

        event = 0.0
        if self._cooldown > 0:
            self._cooldown -= 1
            active = False

        if active and (not self._last_active):
            event = 1.0
            self._cooldown = self.cooldown_frames

        self._last_active = active

        if self.debug:
            print(f"[CheckpointDetector] ratio={ratio:.3f} active={int(active)} event={int(event)}")

        return event, ratio
