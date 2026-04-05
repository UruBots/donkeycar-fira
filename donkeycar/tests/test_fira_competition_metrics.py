import pytest

from donkeycar.parts.fira_competition_metrics import FiraCompetitionMetrics


def test_right_lane_score_and_violations_accumulate_on_active_frames():
    part = FiraCompetitionMetrics(
        enabled=True,
        lane_conf_min=0.5,
        right_lane_angle_min=-0.25,
        right_lane_angle_max=0.05,
        checkpoint_min_right_lane_streak=4,
        checkpoint_cooldown_frames=2,
        target_checkpoints=5,
    )

    # 3 compliant frames in active state
    for _ in range(3):
        score, checkpoints, progress, violations, active, compliance, ready = part.run(
            lane_confidence=0.9,
            lane_angle=-0.10,
            drive_state='race_fast',
            obstacle_severity=0.1,
        )

    assert active == 3
    assert violations == 0
    assert score == pytest.approx(1.0)
    assert checkpoints == 0
    assert progress == pytest.approx(0.0)
    assert 0.0 <= compliance <= 1.0
    assert ready is False

    # 1 violating frame (low confidence)
    score, checkpoints, progress, violations, active, compliance, ready = part.run(
        lane_confidence=0.2,
        lane_angle=-0.10,
        drive_state='race_fast',
        obstacle_severity=0.1,
    )

    assert active == 4
    assert violations == 1
    assert checkpoints == 0
    assert score == pytest.approx(3.0 / 4.0)
    assert progress == pytest.approx(0.0)
    assert 0.0 <= compliance <= 1.0
    assert ready is False


def test_checkpoint_count_uses_streak_and_cooldown():
    part = FiraCompetitionMetrics(
        enabled=True,
        lane_conf_min=0.4,
        right_lane_angle_min=-0.30,
        right_lane_angle_max=0.05,
        checkpoint_min_right_lane_streak=3,
        checkpoint_cooldown_frames=2,
        target_checkpoints=4,
    )

    # First checkpoint by streak
    for _ in range(3):
        score, checkpoints, progress, violations, active, compliance, ready = part.run(
            lane_confidence=0.8,
            lane_angle=-0.15,
            drive_state='urban_navigate',
            obstacle_severity=0.0,
        )

    assert checkpoints == 1
    assert progress == pytest.approx(0.25)

    # Cooldown should block immediate next checkpoint
    for _ in range(2):
        score, checkpoints, progress, violations, active, compliance, ready = part.run(
            lane_confidence=0.8,
            lane_angle=-0.15,
            drive_state='urban_navigate',
            obstacle_severity=0.0,
        )

    assert checkpoints == 1

    # New checkpoint after cooldown + new streak
    for _ in range(3):
        score, checkpoints, progress, violations, active, compliance, ready = part.run(
            lane_confidence=0.8,
            lane_angle=-0.15,
            drive_state='urban_navigate',
            obstacle_severity=0.0,
        )

    assert checkpoints == 2
    assert progress == pytest.approx(0.5)
    assert 0.0 <= compliance <= 1.0
    assert isinstance(ready, bool)


def test_disabled_metrics_returns_zeroes():
    part = FiraCompetitionMetrics(enabled=False)

    out = part.run(
        lane_confidence=1.0,
        lane_angle=0.0,
        drive_state='race_fast',
        obstacle_severity=0.0,
    )

    assert out == (0.0, 0, 0.0, 0, 0, 0.0, False)


def test_real_checkpoint_event_overrides_proxy_and_counts_on_rising_edge():
    part = FiraCompetitionMetrics(
        enabled=True,
        lane_conf_min=0.4,
        right_lane_angle_min=-0.30,
        right_lane_angle_max=0.10,
        checkpoint_min_right_lane_streak=10,
        checkpoint_cooldown_frames=0,
        target_checkpoints=4,
    )

    # First rising-edge event counts checkpoint.
    out = part.run(
        lane_confidence=0.9,
        lane_angle=-0.10,
        drive_state='urban_navigate',
        obstacle_severity=0.0,
        checkpoint_event=1.0,
    )
    assert out[1] == 1

    # Sustained high signal should not double-count until falling edge.
    out = part.run(
        lane_confidence=0.9,
        lane_angle=-0.10,
        drive_state='urban_navigate',
        obstacle_severity=0.0,
        checkpoint_event=1.0,
    )
    assert out[1] == 1

    # New rising edge after low state increments again.
    part.run(
        lane_confidence=0.9,
        lane_angle=-0.10,
        drive_state='urban_navigate',
        obstacle_severity=0.0,
        checkpoint_event=0.0,
    )
    out = part.run(
        lane_confidence=0.9,
        lane_angle=-0.10,
        drive_state='urban_navigate',
        obstacle_severity=0.0,
        checkpoint_event=1.0,
    )
    assert out[1] == 2


def test_checkpoint_event_outside_right_lane_adds_violation():
    part = FiraCompetitionMetrics(
        enabled=True,
        lane_conf_min=0.5,
        right_lane_angle_min=-0.25,
        right_lane_angle_max=0.05,
        checkpoint_min_right_lane_streak=10,
        checkpoint_cooldown_frames=0,
        target_checkpoints=5,
    )

    # Not right-lane compliant at checkpoint crossing.
    score, checkpoints, progress, violations, *_ = part.run(
        lane_confidence=0.9,
        lane_angle=0.35,
        drive_state='urban_navigate',
        obstacle_severity=0.0,
        checkpoint_event=1.0,
    )

    assert checkpoints == 1
    assert violations >= 1
