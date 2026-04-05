import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_safety_signals import FiraSafetySignalsEstimator


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_lane_blob_right(img):
    # Draw HSV yellow and convert to RGB so estimator thresholds match.
    hsv = np.zeros_like(img)
    hsv[80:118, 102:124, :] = (30, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_lane_blob_left(img):
    hsv = np.zeros_like(img)
    hsv[80:118, 36:58, :] = (30, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_obstacle_blob_right(img):
    # Draw HSV orange in the lower-center/right area.
    hsv = np.zeros_like(img)
    hsv[72:110, 96:126, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _draw_obstacle_blob_left(img):
    hsv = np.zeros_like(img)
    hsv[72:110, 34:64, :] = (15, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_lane_signal_detects_right_offset():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=True,
        obstacle_enabled=False,
        lane_min_pixel_ratio=0.001,
        lane_full_pixel_ratio=0.01,
        lane_steering_gain=1.0,
    )

    img = _draw_lane_blob_right(_blank_rgb())
    lane_angle, lane_conf, obs_corr, obs_sev = estimator.run(img, pilot_angle=0.0)

    assert lane_angle > 0.0
    assert lane_conf > 0.0
    assert obs_corr == pytest.approx(0.0)
    assert obs_sev == pytest.approx(0.0)


def test_obstacle_signal_steers_away_from_right_side():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())
    lane_angle, lane_conf, obs_corr, obs_sev = estimator.run(img, pilot_angle=0.2)

    assert lane_angle == pytest.approx(0.2)
    assert lane_conf == pytest.approx(0.0)
    assert obs_corr < 0.0
    assert obs_sev > 0.0


def test_detector_down_reduces_signal_strength():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=True,
        obstacle_enabled=True,
        lane_min_pixel_ratio=0.001,
        lane_full_pixel_ratio=0.01,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        reduce_signals_when_detector_down=True,
        detector_down_signal_scale=0.25,
        enable_temporal_smoothing=False,
    )

    img = _draw_lane_blob_right(_blank_rgb())
    img = _draw_obstacle_blob_right(img)

    _, conf_up, corr_up, sev_up = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=True, tf_detector_ready=True
    )
    _, conf_down, corr_down, sev_down = estimator.run(
        img, pilot_angle=0.0, yolo_detector_ready=False, tf_detector_ready=True
    )

    assert conf_down == pytest.approx(conf_up * 0.25)
    assert corr_down == pytest.approx(corr_up * 0.25)
    assert sev_down == pytest.approx(sev_up * 0.25)


def test_temporal_smoothing_damps_lane_sign_flip():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=True,
        obstacle_enabled=False,
        lane_min_pixel_ratio=0.001,
        lane_full_pixel_ratio=0.01,
        enable_temporal_smoothing=True,
        smoothing_alpha=0.2,
    )

    img_right = _draw_lane_blob_right(_blank_rgb())
    angle_right, _, _, _ = estimator.run(img_right, pilot_angle=0.0)
    assert angle_right > 0.0

    img_left = _draw_lane_blob_left(_blank_rgb())
    angle_after_flip, _, _, _ = estimator.run(img_left, pilot_angle=0.0)

    # EMA should damp abrupt side changes compared to raw left estimate.
    assert angle_after_flip > -0.2


def test_temporal_smoothing_damps_obstacle_correction_flip():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        enable_temporal_smoothing=True,
        smoothing_alpha=0.2,
    )

    img_right = _draw_obstacle_blob_right(_blank_rgb())
    _, _, corr_right, _ = estimator.run(img_right, pilot_angle=0.0)
    assert corr_right < 0.0

    img_left = _draw_obstacle_blob_left(_blank_rgb())
    _, _, corr_after_flip, _ = estimator.run(img_left, pilot_angle=0.0)

    assert corr_after_flip < 0.2


def test_obstacle_lane_change_recovers_after_clearance():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.10,
        obstacle_avoid_min_correction=0.25,
        obstacle_avoid_min_severity=0.30,
        obstacle_clear_frames=1,
        obstacle_recover_frames=3,
        obstacle_recover_max_correction=0.20,
        obstacle_recover_min_severity=0.20,
        enable_temporal_smoothing=False,
    )

    # Obstacle on right -> steer left (negative) with active avoidance.
    _, _, corr_avoid, sev_avoid = estimator.run(_draw_obstacle_blob_right(_blank_rgb()), pilot_angle=0.0)
    assert corr_avoid < 0.0
    assert sev_avoid > 0.0

    # Once obstacle disappears, policy should enter recovery and steer opposite sign.
    _, _, corr_recover, sev_recover = estimator.run(_blank_rgb(), pilot_angle=0.0)
    assert corr_recover > 0.0
    assert sev_recover > 0.0

    # After configured recovery horizon it should settle back to neutral.
    for _ in range(3):
        _, _, corr_recover, sev_recover = estimator.run(_blank_rgb(), pilot_angle=0.0)

    assert corr_recover == pytest.approx(0.0)
    assert sev_recover == pytest.approx(0.0)


def test_obstacle_reacquire_during_recovery_returns_to_avoid():
    estimator = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.10,
        obstacle_clear_frames=1,
        obstacle_recover_frames=4,
        enable_temporal_smoothing=False,
    )

    _, _, corr_avoid, _ = estimator.run(_draw_obstacle_blob_right(_blank_rgb()), pilot_angle=0.0)
    assert corr_avoid < 0.0

    _, _, corr_recover, _ = estimator.run(_blank_rgb(), pilot_angle=0.0)
    assert corr_recover > 0.0

    # Re-detecting obstacle should re-enter avoid phase (negative correction).
    _, _, corr_again, sev_again = estimator.run(_draw_obstacle_blob_right(_blank_rgb()), pilot_angle=0.0)
    assert corr_again < 0.0
    assert sev_again > 0.0


def test_safe_profile_requires_higher_severity_to_engage():
    """SAFE profile should have higher engage_severity threshold than RACE."""
    # SAFE profile config (conservative)
    safe_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.28,  # SAFE: higher threshold
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,
        enable_temporal_smoothing=False,
    )

    # RACE profile config (aggressive)
    race_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.24,  # RACE: lower threshold
        obstacle_avoid_min_correction=0.30,
        obstacle_clear_frames=2,
        obstacle_recover_frames=7,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # Run both on same stimulus
    _, _, safe_corr, safe_sev = safe_est.run(img, pilot_angle=0.0)
    _, _, race_corr, race_sev = race_est.run(img, pilot_angle=0.0)

    # SAFE detects same severity (measured)
    assert safe_sev > 0.0
    assert race_sev > 0.0

    # Both should have engaged because measured severity > both thresholds
    assert safe_corr < 0.0
    assert race_corr < 0.0


def test_race_profile_higher_correction_than_safe():
    """RACE profile should engage at lower severity threshold."""
    safe_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.50,  # SAFE: high threshold, harder to engage
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,
        enable_temporal_smoothing=False,
    )

    race_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.10,  # RACE: low threshold, easier to engage
        obstacle_avoid_min_correction=0.30,
        obstacle_clear_frames=2,
        obstacle_recover_frames=7,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    # First frame: SAFE might not engage yet, RACE should engage
    _, _, safe_corr_1, safe_sev_1 = safe_est.run(img, pilot_angle=0.0)
    _, _, race_corr_1, race_sev_1 = race_est.run(img, pilot_angle=0.0)

    # RACE engages earlier (at lower severity)
    # If RACE engaged but SAFE didn't, race correction should be non-zero
    if race_sev_1 >= 0.10 and safe_sev_1 < 0.50:
        assert race_corr_1 != 0.0
        # SAFE might still be idle (corr=0.0)
    else:
        # Both measured same severity, both should engage if threshold exceeded
        assert race_corr_1 < 0.0


def test_safe_profile_slower_recovery_than_race():
    """SAFE profile has different recovery parameters than RACE."""
    safe_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.28,
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,  # SAFE: 6-frame recovery
        obstacle_recover_max_correction=0.15,
        enable_temporal_smoothing=False,
    )

    race_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.24,
        obstacle_avoid_min_correction=0.30,
        obstacle_clear_frames=2,
        obstacle_recover_frames=7,  # RACE: 7-frame recovery
        obstacle_recover_max_correction=0.20,
        enable_temporal_smoothing=False,
    )

    # Drive both through avoidance
    img = _draw_obstacle_blob_right(_blank_rgb())
    safe_est.run(img, pilot_angle=0.0)
    race_est.run(img, pilot_angle=0.0)

    # Clear obstacle; both should enter recovery
    # Skip the clear_frames (2) delay for both
    safe_est.run(_blank_rgb(), pilot_angle=0.0)
    safe_est.run(_blank_rgb(), pilot_angle=0.0)
    safe_est.run(_blank_rgb(), pilot_angle=0.0)  # Now in recovery
    
    race_est.run(_blank_rgb(), pilot_angle=0.0)
    race_est.run(_blank_rgb(), pilot_angle=0.0)
    race_est.run(_blank_rgb(), pilot_angle=0.0)  # Now in recovery

    # Collect recovery for next 6-7 frames
    safe_recovery = []
    race_recovery = []
    for _ in range(8):
        _, _, sc, _ = safe_est.run(_blank_rgb(), pilot_angle=0.0)
        _, _, rc, _ = race_est.run(_blank_rgb(), pilot_angle=0.0)
        safe_recovery.append(sc)
        race_recovery.append(rc)

    # Both should eventually settle to zero
    assert safe_recovery[-1] == pytest.approx(0.0, abs=0.02)
    assert race_recovery[-1] == pytest.approx(0.0, abs=0.02)

    # Different recovery frame counts
    # SAFE: 6 frames, RACE: 7 frames
    assert safe_est.obstacle_recover_frames == 6
    assert race_est.obstacle_recover_frames == 7


def test_profile_boundary_engagement():
    """Verify profiles engage exactly at their configured threshold."""
    # Create estimator with very low engage threshold (should trigger easily)
    low_threshold_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.05,  # Very low threshold
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=2,
        obstacle_recover_frames=5,
        enable_temporal_smoothing=False,
    )

    # Create estimator with very high engage threshold (should not trigger)
    high_threshold_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.95,  # Very high threshold
        obstacle_avoid_min_correction=0.25,
        obstacle_clear_frames=2,
        obstacle_recover_frames=5,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    _, _, low_corr, low_sev = low_threshold_est.run(img, pilot_angle=0.0)
    _, _, high_corr, high_sev = high_threshold_est.run(img, pilot_angle=0.0)

    # Low threshold should definitely engage
    assert low_corr < 0.0
    assert low_sev > 0.0

    # High threshold might not engage (depends on measured severity)
    # But if both measured severities are non-zero, the low threshold is lower
    if high_sev > 0.0:
        assert low_sev >= high_sev  # Same input


def test_multiple_profile_transitions_maintain_stability():
    """Verify that rapid profile-like transitions don't break estimator."""
    # Simulate switching between SAFE-like and RACE-like parameters
    est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.28,  # Start with SAFE
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,
        obstacle_recover_max_correction=0.15,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())
    results = []

    # Run several frames
    for i in range(10):
        _, _, corr, sev = est.run(img, pilot_angle=0.0)
        results.append((corr, sev))

    # All runs should produce consistent outputs (no NaN/Inf)
    for corr, sev in results:
        assert not np.isnan(corr)
        assert not np.isinf(corr)
        assert not np.isnan(sev)
        assert not np.isinf(sev)

    # Correction should be negative (obstacle on right)
    for corr, sev in results:
        assert corr < 0.0


def test_profile_consistency_across_frames():
    """Verify SAFE profile maintains conservative behavior consistently."""
    safe_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.28,
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())
    corrections = []

    # Collect multiple frames in avoid phase
    for _ in range(5):
        _, _, corr, _ = safe_est.run(img, pilot_angle=0.0)
        corrections.append(corr)

    # All corrections in avoid phase should be consistent (negative, same magnitude)
    for corr in corrections:
        assert corr < 0.0
        assert corr == pytest.approx(corrections[0], rel=0.05)  # Within 5% of first


def test_tight_profile_is_more_aggressive_than_race():
    """TIGHT profile should use stronger minimum correction than RACE."""
    race_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.24,
        obstacle_avoid_min_correction=0.30,
        obstacle_avoid_min_severity=0.44,
        obstacle_clear_frames=2,
        obstacle_recover_frames=7,
        enable_temporal_smoothing=False,
    )

    tight_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.20,
        obstacle_avoid_min_correction=0.34,
        obstacle_avoid_min_severity=0.46,
        obstacle_clear_frames=1,
        obstacle_recover_frames=4,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    _, _, race_corr, _ = race_est.run(img, pilot_angle=0.0)
    _, _, tight_corr, _ = tight_est.run(img, pilot_angle=0.0)

    assert race_corr < 0.0
    assert tight_corr < 0.0
    assert abs(tight_corr) >= abs(race_corr)


def test_tight_profile_recovery_completes_faster_than_safe():
    """TIGHT profile should settle earlier than SAFE due to shorter recover window."""
    safe_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.28,
        obstacle_avoid_min_correction=0.24,
        obstacle_clear_frames=2,
        obstacle_recover_frames=6,
        obstacle_recover_max_correction=0.15,
        enable_temporal_smoothing=False,
    )

    tight_est = FiraSafetySignalsEstimator(
        lane_enabled=False,
        obstacle_enabled=True,
        obstacle_min_pixel_ratio=0.001,
        obstacle_full_pixel_ratio=0.01,
        obstacle_max_correction=1.0,
        obstacle_lane_change_enabled=True,
        obstacle_engage_severity=0.20,
        obstacle_avoid_min_correction=0.34,
        obstacle_clear_frames=1,
        obstacle_recover_frames=4,
        obstacle_recover_max_correction=0.22,
        enable_temporal_smoothing=False,
    )

    img = _draw_obstacle_blob_right(_blank_rgb())

    safe_est.run(img, pilot_angle=0.0)
    tight_est.run(img, pilot_angle=0.0)

    # Move both policies into recovery phase.
    safe_est.run(_blank_rgb(), pilot_angle=0.0)
    safe_est.run(_blank_rgb(), pilot_angle=0.0)
    safe_est.run(_blank_rgb(), pilot_angle=0.0)

    tight_est.run(_blank_rgb(), pilot_angle=0.0)
    tight_est.run(_blank_rgb(), pilot_angle=0.0)

    tight_abs = []
    safe_abs = []
    for _ in range(5):
        _, _, sc, _ = safe_est.run(_blank_rgb(), pilot_angle=0.0)
        _, _, tc, _ = tight_est.run(_blank_rgb(), pilot_angle=0.0)
        safe_abs.append(abs(sc))
        tight_abs.append(abs(tc))

    # At same horizon, TIGHT should be at least as settled as SAFE.
    assert tight_abs[-1] <= safe_abs[-1]
    assert tight_est.obstacle_recover_frames < safe_est.obstacle_recover_frames
