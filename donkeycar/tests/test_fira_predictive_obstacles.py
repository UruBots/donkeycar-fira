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


def _scene_with_cone(x0, x1, y0=72, y1=112):
    img = _draw_lane_corridor(_blank_rgb())
    return _draw_cone(img, x0, x1, y0=y0, y1=y1)


def _part(predictive_enabled=False, predictive_min_confidence=0.25):
    return FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=5000.0,
        predictive_enabled=predictive_enabled,
        predictive_horizon_sec=0.20,
        predictive_blend=0.75,
        predictive_velocity_alpha=0.6,
        predictive_min_confidence=predictive_min_confidence,
        predictive_max_missing_frames=2,
        debug=False,
    )


def test_predictive_blending_anticipates_obstacle_motion():
    base = _part(predictive_enabled=False)
    pred = _part(predictive_enabled=True, predictive_min_confidence=0.20)

    frame_1 = _scene_with_cone(112, 128)
    frame_2 = _scene_with_cone(98, 116)

    base.run(0.0, 0.30, frame_1)
    pred.run(0.0, 0.30, frame_1)

    base_angle_2, _, _ = base.run(0.0, 0.30, frame_2)
    pred_angle_2, _, _ = pred.run(0.0, 0.30, frame_2)

    # Moving obstacle drifts toward lane center; prediction should soften
    # current-frame correction versus purely reactive behavior.
    assert base_angle_2 < 0.0
    assert pred_angle_2 < 0.0
    assert abs(pred_angle_2) < abs(base_angle_2)


def test_predictive_fallback_decays_when_obstacle_disappears():
    pred = _part(predictive_enabled=True, predictive_min_confidence=0.20)

    frame_obs = _scene_with_cone(108, 126)
    frame_clear = _draw_lane_corridor(_blank_rgb())

    pred.run(0.0, 0.30, frame_obs)
    pred.run(0.0, 0.30, frame_obs)

    angle = 0.0
    for _ in range(8):
        angle, _, _ = pred.run(0.0, 0.30, frame_clear)

    assert abs(angle) < 0.08


def test_low_confidence_gate_prevents_wrong_side_prediction():
    pred = _part(predictive_enabled=True, predictive_min_confidence=0.65)

    frame_right = _scene_with_cone(108, 126)
    frame_left = _scene_with_cone(36, 54)

    pred.run(0.0, 0.30, frame_right)
    out_angle, _, _ = pred.run(0.0, 0.30, frame_left)

    # Large jump should not trust stale velocity prediction; current detection wins.
    assert out_angle > 0.0
