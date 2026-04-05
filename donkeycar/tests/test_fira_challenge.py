import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_challenge import FiraChallengePart


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_lane_corridor(img):
    hsv = np.zeros_like(img)
    # Left and right lane boundaries in yellow.
    hsv[:, 24:30, :] = (30, 255, 255)
    hsv[:, 130:136, :] = (30, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_cone_right(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[72:112, 102:122, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_cone_left(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[72:112, 40:60, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_dark_car_right(img):
    # Dark gray car blob matched by car HSV threshold.
    rgb = img.copy()
    rgb[70:108, 96:130, :] = (55, 55, 55)
    return rgb


def _base_scene():
    return _draw_lane_corridor(_blank_rgb())


def test_steers_left_when_obstacle_is_on_right():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )
    img = _draw_cone_right(_base_scene())

    out_angle, out_throttle, _ = part.run(0.0, 0.30, img)

    assert out_angle < 0.0
    assert out_throttle < 0.30


def test_steers_right_when_obstacle_is_on_left():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )
    img = _draw_cone_left(_base_scene())

    out_angle, out_throttle, _ = part.run(0.0, 0.30, img)

    assert out_angle > 0.0
    assert out_throttle < 0.30


def test_handles_sequential_s_avoidance_for_two_obstacles():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        clear_frames=1,
        recover_frames=2,
        smoothing_alpha=0.8,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )

    img_right = _draw_cone_right(_base_scene())
    angle1, _, _ = part.run(0.0, 0.30, img_right)
    assert angle1 < 0.0

    # Clear one frame to trigger recover behavior.
    part.run(0.0, 0.30, _base_scene())

    img_left = _draw_cone_left(_base_scene())
    angle2, _, _ = part.run(0.0, 0.30, img_left)

    # Must re-plan and flip sign when second obstacle appears on the opposite side.
    assert angle2 > 0.0


def test_detects_dark_car_blob_and_applies_evasion():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )
    img = _draw_dark_car_right(_base_scene())

    out_angle, out_throttle, _ = part.run(0.0, 0.30, img)

    assert out_angle < 0.0
    assert out_throttle < 0.30


def test_lane_confidence_hint_reduces_correction_when_uncertain():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        debug=False,
    )
    img = _draw_cone_right(_base_scene())

    strong_angle, _, _ = part.run(0.0, 0.30, img, lane_confidence=1.0)
    weak_angle, _, _ = part.run(0.0, 0.30, img, lane_confidence=0.0)

    assert abs(weak_angle) < abs(strong_angle)


def test_warmup_temporarily_reduces_initial_obstacle_response():
    no_warmup = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        warmup_frames=0,
        debug=False,
    )
    with_warmup = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        warmup_frames=10,
        warmup_steering_gain=0.18,
        warmup_max_abs_angle=0.25,
        warmup_throttle=0.26,
        debug=False,
    )

    img = _draw_cone_right(_base_scene())

    angle_no_warmup, throttle_no_warmup, _ = no_warmup.run(0.0, 0.30, img)
    angle_warmup, throttle_warmup, _ = with_warmup.run(0.0, 0.30, img)

    # Warmup should be gentler at startup than full obstacle authority.
    assert abs(angle_warmup) < abs(angle_no_warmup)
    assert throttle_warmup <= 0.26
    assert throttle_warmup <= 0.30

    # After warmup horizon, behavior should approach regular obstacle response.
    for _ in range(12):
        angle_late, _, _ = with_warmup.run(0.0, 0.30, img)
    assert abs(angle_late) >= abs(angle_warmup)


def test_imminent_collision_caps_throttle_for_safety():
    part = FiraChallengePart(
        enabled=True,
        obstacle_min_area_ratio=0.001,
        obstacle_full_area_ratio=0.01,
        engage_severity=0.05,
        smoothing_alpha=1.0,
        max_steering_delta_per_sec=20.0,
        collision_imminent_enabled=True,
        collision_imminent_proximity=0.80,
        collision_imminent_occupancy=0.30,
        collision_imminent_throttle_cap=0.015,
        debug=False,
    )

    # Big close obstacle to trigger imminent-collision guard.
    img = _base_scene()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[68:110, 62:124, :] = (15, 255, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    _, out_throttle, _ = part.run(0.0, 0.30, img)

    assert out_throttle <= 0.015
