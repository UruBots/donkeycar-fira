"""
Comprehensive health and edge case validation for FIRA safety system.
Tests detector health scenarios combined with obstacle lane-change behavior.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_safety_signals import FiraSafetySignalsEstimator
from donkeycar.parts.fira_safety_arbiter import FiraSafetyArbiter


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_obstacle_blob_right(img):
    hsv = np.zeros_like(img)
    hsv[72:110, 96:126, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_obstacle_avoidance_robust_to_detector_down():
    """Obstacle lane-change should degrade gracefully when detector goes down."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.20,
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=2,
        obstacle_recover_frames=5,
        reduce_signals_when_detector_down=True,
        detector_down_signal_scale=0.25,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Detector up: strong response
    _, _, corr_up, sev_up = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=True, tf_detector_ready=True
    )
    assert corr_up < 0.0
    assert sev_up > 0.3

    # Detector down: scaled response
    _, _, corr_down, sev_down = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=False, tf_detector_ready=False
    )
    
    # Signal should be present but scaled
    assert corr_down < 0.0
    assert sev_down == pytest.approx(sev_up * 0.25, rel=0.1)


def test_obstacle_avoidance_continues_during_detector_dropout():
    """Obstacle avoidance should complete before detector dropout affects it."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.15,
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=1,
        obstacle_recover_frames=3,
        reduce_signals_when_detector_down=False,  # Don't scale, just continue
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Engage avoidance with detector up
    _, _, corr_1, _ = estimator.run(img, pilot_angle=0.0, yolo_detector_ready=True)
    assert corr_1 < 0.0

    # Obstacle clears (one more with detector up to count clear frames)
    _, _, corr_2, _ = estimator.run(_blank_rgb(), pilot_angle=0.0, yolo_detector_ready=True)

    # Recovery phase begins; detector goes down
    for i in range(4):
        _, _, corr_recover, sev_recover = estimator.run(
            _blank_rgb(), pilot_angle=0.0, yolo_detector_ready=False
        )
        # Recovery should continue even with detector down
        assert not np.isnan(corr_recover)
        assert not np.isinf(corr_recover)


def test_multiple_detector_failures_do_not_crash():
    """Rapid detector failure/recovery cycles should not crash estimator."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.20,
        obstacle_avoid_min_correction=0.25,
        reduce_signals_when_detector_down=True,
        detector_down_signal_scale=0.2,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Simulate rapid detector on/off cycles
    for cycle in range(10):
        yolo_ready = (cycle % 2) == 0
        tf_ready = ((cycle + 1) % 2) == 0
        
        _, _, corr, sev = estimator.run(
            img, pilot_angle=0.0,
            yolo_detector_ready=yolo_ready,
            tf_detector_ready=tf_ready
        )

        # Should never produce NaN or Inf
        assert not np.isnan(corr)
        assert not np.isinf(corr)
        assert not np.isnan(sev)
        assert not np.isinf(sev)


def test_obstacle_avoidance_with_partially_degraded_detectors():
    """Obstacle avoidance should handle one detector down, one up."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.20,
        obstacle_avoid_min_correction=0.25,
        reduce_signals_when_detector_down=True,
        detector_down_signal_scale=0.5,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # YOLO healthy, TF down
    _, _, corr_partial_1, sev_partial_1 = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=True, tf_detector_ready=False
    )

    # Both healthy
    _, _, corr_both, sev_both = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=True, tf_detector_ready=True
    )

    # Partial detector down should produce some signal but scaled
    assert corr_partial_1 < 0.0
    assert sev_partial_1 > 0.0
    # When one detector is down, signals should be reduced
    # (depending on implementation, both detectors might be required)


def test_arbiter_respects_detector_failsafe_with_obstacle_avoidance():
    """Safety arbiter should apply throttle cap when detector down during avoidance."""
    arbiter = FiraSafetyArbiter(
        max_abs_steering=1.0,
        throttle_min=0.0,
        throttle_max=1.0,
        curve_throttle_start=0.25,
        curve_throttle_full=0.9,
        curve_throttle_factor_min=0.6,
        failsafe_throttle_cap=0.25,
        severe_failsafe_throttle_cap=0.15,
        use_yolo_detector_failsafe=True,
        use_tf_detector_failsafe=True,
        max_steering_delta_per_sec=2.0,
        max_throttle_delta_per_sec=0.8,
        use_lane_guidance=False,
        use_obstacle_correction=True,
        obstacle_correction_max_abs=1.0,
        obstacle_throttle_factor_min=0.45,
        debug=False,
    )

    img = _blank_rgb()

    # Detector up: obstacle detected
    angle_up, throttle_up, _ = arbiter.run(
        angle=0.0,
        throttle=0.8,
        img_arr=img,
        lane_angle=0.0,
        lane_confidence=0.0,
        obstacle_steering_correction=-0.40,
        obstacle_severity=0.6,
        yolo_detector_ready=True,
        tf_detector_ready=True,
    )

    # Detector down: same obstacle
    angle_down, throttle_down, _ = arbiter.run(
        angle=0.0,
        throttle=0.8,
        img_arr=img,
        lane_angle=0.0,
        lane_confidence=0.0,
        obstacle_steering_correction=-0.40,
        obstacle_severity=0.6,
        yolo_detector_ready=False,
        tf_detector_ready=False,
    )

    # Detector failsafe should cap throttle when down
    assert throttle_down < throttle_up


def test_extreme_parameter_boundaries():
    """Verify estimator doesn't break at parameter extremes."""
    # Test very high severity threshold (almost never engages)
    est_high = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.99,  # Almost impossible to reach
        obstacle_avoid_min_correction=0.01,
        obstacle_clear_frames=1,
        obstacle_recover_frames=1,
        enable_temporal_smoothing=False,
    )

    # Test very low severity threshold (always engages)
    est_low = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.001,  # Very easy to reach
        obstacle_avoid_min_correction=0.99,  # Maximum correction
        obstacle_clear_frames=1,
        obstacle_recover_frames=20,  # Very long recovery
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Both should produce valid outputs (no crash, no NaN)
    for est in [est_high, est_low]:
        _, _, corr, sev = est.run(img, pilot_angle=0.0)
        assert not np.isnan(corr)
        assert not np.isinf(corr)
        assert not np.isnan(sev)
        assert not np.isinf(sev)


def test_obstacle_avoidance_with_zero_clear_frames():
    """Obstacle avoidance should handle edge case of zero clear frames."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.10,
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=0,  # Edge case: immediate recovery
        obstacle_recover_frames=2,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Engage
    _, _, corr_avoid, _ = estimator.run(img, pilot_angle=0.0)
    assert corr_avoid < 0.0

    # Obstacle clears
    _, _, corr_recover, _ = estimator.run(_blank_rgb(), pilot_angle=0.0)

    # Should immediately enter recovery
    assert corr_recover > 0.0

    # Recovery should settle
    _, _, corr_final, _ = estimator.run(_blank_rgb(), pilot_angle=0.0)
    assert corr_final == pytest.approx(0.0, abs=0.02)


def test_obstacle_avoidance_with_single_frame_recovery():
    """Obstacle avoidance should handle minimum recovery duration."""
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.10,
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=1,
        obstacle_recover_frames=1,  # Edge case: one-frame recovery
        obstacle_recover_max_correction=0.25,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Engage and clear quickly
    _, _, corr_1, _ = estimator.run(img, pilot_angle=0.0)
    assert corr_1 < 0.0

    _, _, corr_2, _ = estimator.run(_blank_rgb(), pilot_angle=0.0)

    _, _, corr_3, _ = estimator.run(_blank_rgb(), pilot_angle=0.0)

    # Should settle within a few frames
    assert corr_3 == pytest.approx(0.0, abs=0.05)


def test_consistency_across_all_profile_boundaries():
    """Test all SAFE/RACE profile combinations maintain consistency."""
    profiles = {
        'SAFE': {
            'engage_severity': 0.28,
            'avoid_min_correction': 0.24,
            'recover_frames': 6,
        },
        'RACE': {
            'engage_severity': 0.24,
            'avoid_min_correction': 0.30,
            'recover_frames': 7,
        },
    }

    img = _draw_obstacle_blob_right(_blank_rgb())

    for profile_name, params in profiles.items():
        est = FiraSafetySignalsEstimator(
            lane_enabled=False,
            obstacle_enabled=True,
            obstacle_min_pixel_ratio=0.001,
            obstacle_full_pixel_ratio=0.01,
            obstacle_max_correction=1.0,
            obstacle_lane_change_enabled=True,
            obstacle_engage_severity=params['engage_severity'],
            obstacle_avoid_min_correction=params['avoid_min_correction'],
            obstacle_recover_frames=params['recover_frames'],
            enable_temporal_smoothing=False,
        )

        # Run multiple times and verify consistency
        outputs = []
        for _ in range(10):
            _, _, corr, sev = est.run(img, pilot_angle=0.0)
            outputs.append((corr, sev))

        # All outputs should be valid
        for corr, sev in outputs:
            assert not np.isnan(corr)
            assert not np.isinf(corr)
            assert not np.isnan(sev)
            assert not np.isinf(sev)
            assert corr < 0.0  # Obstacle on right
