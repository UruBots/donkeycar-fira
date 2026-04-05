import numpy as np
import pytest

from donkeycar.parts import fira_safety_arbiter as safety_mod


def _img():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_curve_speed_reduction_applies():
    arbiter = safety_mod.FiraSafetyArbiter(
        curve_throttle_start=0.2,
        curve_throttle_full=0.8,
        curve_throttle_factor_min=0.5,
    )

    angle, throttle, _ = arbiter.run(0.8, 0.4, _img())

    assert angle == 0.8
    assert throttle == 0.2


def test_detector_failsafe_caps_throttle():
    arbiter = safety_mod.FiraSafetyArbiter(
        failsafe_throttle_cap=0.07,
        use_yolo_detector_failsafe=True,
    )

    _, throttle, _ = arbiter.run(0.0, 0.5, _img(), yolo_detector_ready=False)

    assert throttle == 0.07


def test_detector_failsafe_uses_stricter_cap_for_backend_unavailable():
    arbiter = safety_mod.FiraSafetyArbiter(
        failsafe_throttle_cap=0.07,
        severe_failsafe_throttle_cap=0.03,
        use_yolo_detector_failsafe=True,
    )

    _, throttle, _ = arbiter.run(
        0.0,
        0.5,
        _img(),
        yolo_detector_health={
            'detector_ready': False,
            'detector_backend': 'unavailable',
            'last_detection_error': 'backend unavailable',
        },
    )

    assert throttle == 0.03


def test_slew_rate_limits_output(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr(safety_mod.time, 'time', _time)

    arbiter = safety_mod.FiraSafetyArbiter(
        max_steering_delta_per_sec=0.5,
        max_throttle_delta_per_sec=0.1,
    )

    # Initialize internal state.
    arbiter.run(0.0, 0.0, _img())

    # One second later, request a large change.
    fake_time['value'] = 11.0
    angle, throttle, _ = arbiter.run(1.0, 1.0, _img())

    assert angle == 0.5
    assert throttle == 0.1


def test_lane_confidence_blend_applies_when_enabled():
    arbiter = safety_mod.FiraSafetyArbiter(
        use_lane_guidance=True,
        lane_confidence_min=0.3,
        lane_blend_max=1.0,
        lane_angle_max_abs=1.0,
        curve_throttle_start=1.0,
        curve_throttle_full=1.0,
    )

    angle, throttle, _ = arbiter.run(
        angle=0.2,
        throttle=0.3,
        img_arr=_img(),
        lane_angle=0.8,
        lane_confidence=1.0,
    )

    assert angle == pytest.approx(0.8)
    assert throttle == pytest.approx(0.3)


def test_obstacle_correction_has_priority_and_reduces_speed():
    arbiter = safety_mod.FiraSafetyArbiter(
        use_lane_guidance=True,
        lane_confidence_min=0.3,
        lane_blend_max=1.0,
        use_obstacle_correction=True,
        obstacle_correction_max_abs=1.0,
        obstacle_blend_max=1.0,
        obstacle_throttle_factor_min=0.4,
        curve_throttle_start=1.0,
        curve_throttle_full=1.0,
    )

    angle, throttle, _ = arbiter.run(
        angle=0.0,
        throttle=0.5,
        img_arr=_img(),
        lane_angle=0.8,
        lane_confidence=1.0,
        obstacle_steering_correction=-0.6,
        obstacle_severity=1.0,
    )

    assert angle == pytest.approx(0.2)
    assert throttle == pytest.approx(0.2)


def test_telemetry_adapter_reports_failsafe_and_weights():
    arbiter = safety_mod.FiraSafetyArbiter(
        use_yolo_detector_failsafe=True,
        failsafe_throttle_cap=0.06,
        use_lane_guidance=True,
        lane_confidence_min=0.2,
        lane_blend_max=1.0,
        use_obstacle_correction=True,
        obstacle_blend_max=1.0,
        curve_throttle_start=1.0,
        curve_throttle_full=1.0,
    )
    wrapped = safety_mod.FiraSafetyArbiterTelemetryAdapter(arbiter)

    out = wrapped.run(
        angle=0.0,
        throttle=0.5,
        img_arr=_img(),
        yolo_detector_health={'detector_ready': False, 'detector_backend': 'unavailable'},
        lane_angle=0.5,
        lane_confidence=1.0,
        obstacle_steering_correction=-0.4,
        obstacle_severity=0.5,
    )

    assert len(out) == 8
    safe_angle, safe_throttle, _, failsafe_active, lane_weight, obstacle_weight, curve_factor, speed_factor = out
    assert failsafe_active is True
    assert lane_weight > 0.0
    assert obstacle_weight > 0.0
    assert curve_factor == pytest.approx(1.0)
    assert 0.0 <= speed_factor <= 1.0
    assert safe_throttle <= 0.06
    assert isinstance(safe_angle, float)
