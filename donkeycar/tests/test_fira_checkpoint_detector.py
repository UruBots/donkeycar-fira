import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_checkpoint_detector import FiraCheckpointDetector


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_red_band_bottom(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[92:118, 12:148, :] = (0, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_checkpoint_detector_emits_rising_edge_event():
    det = FiraCheckpointDetector(enabled=True, min_pixel_ratio=0.05, cooldown_frames=0)

    img_clear = _blank_rgb()
    img_checkpoint = _draw_red_band_bottom(_blank_rgb())

    event, conf = det.run(img_clear)
    assert event == 0.0

    event, conf = det.run(img_checkpoint)
    assert event == 1.0
    assert conf > 0.05

    # Same frame pattern should not retrigger without falling edge.
    event, _ = det.run(img_checkpoint)
    assert event == 0.0


def test_checkpoint_detector_cooldown_prevents_double_count():
    det = FiraCheckpointDetector(enabled=True, min_pixel_ratio=0.05, cooldown_frames=2)
    img_checkpoint = _draw_red_band_bottom(_blank_rgb())

    event1, _ = det.run(img_checkpoint)
    event2, _ = det.run(img_checkpoint)
    event3, _ = det.run(img_checkpoint)

    assert event1 == 1.0
    assert event2 == 0.0
    assert event3 == 0.0
