import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_challenge import FiraChallengePart


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_lane_corridor(img):
    hsv = np.zeros_like(img)
    hsv[:, 24:30, :] = (30, 255, 255)
    hsv[:, 130:136, :] = (30, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_cone(img, x0, x1, y0=72, y1=112):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[y0:y1, x0:x1, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _base_scene():
    return _draw_lane_corridor(_blank_rgb())


def _part():
    return FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )


def test_dual_cones_prefers_center_gap_with_small_steering():
    part = _part()
    img = _base_scene()
    img = _draw_cone(img, 44, 62)
    img = _draw_cone(img, 98, 116)

    out_angle, out_throttle, _ = part.run(0.0, 0.30, img)

    assert abs(out_angle) < 0.20
    assert out_throttle < 0.30


def test_staggered_cones_bias_to_safer_right_gap():
    part = _part()
    img = _base_scene()
    # Near-left cone should dominate risk and push target right.
    img = _draw_cone(img, 40, 62, 72, 112)
    # Far-right cone should still be considered but with lower proximity impact.
    img = _draw_cone(img, 104, 118, 50, 80)

    out_angle, _, _ = part.run(0.0, 0.30, img)

    assert out_angle > 0.0


def test_narrowed_pair_selects_wider_right_outer_gap():
    part = _part()
    img = _base_scene()
    # Pair shifted left leaves right outer gap wider than left.
    img = _draw_cone(img, 56, 76)
    img = _draw_cone(img, 80, 100)

    out_angle, _, _ = part.run(0.0, 0.30, img)

    assert out_angle > 0.0
