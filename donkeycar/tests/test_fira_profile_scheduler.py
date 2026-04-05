from donkeycar.parts.fira_profile_scheduler import FiraProfileScheduler


def test_scheduler_starts_with_default_profile():
    scheduler = FiraProfileScheduler(enabled=True, default_profile='SAFE', min_dwell_frames=0)
    assert scheduler.run() == 'SAFE'


def test_scheduler_switches_to_race_then_tight_by_severity():
    scheduler = FiraProfileScheduler(
        enabled=True,
        default_profile='SAFE',
        min_dwell_frames=0,
        race_enter_severity=0.30,
        tight_enter_severity=0.60,
    )

    assert scheduler.run(lane_confidence=0.9, obstacle_severity=0.35) == 'RACE'
    assert scheduler.run(lane_confidence=0.9, obstacle_severity=0.70) == 'TIGHT'


def test_scheduler_hysteresis_prevents_flapping():
    scheduler = FiraProfileScheduler(
        enabled=True,
        default_profile='SAFE',
        min_dwell_frames=0,
        race_enter_severity=0.30,
        race_exit_severity=0.18,
    )

    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.31) == 'RACE'
    # Stay in RACE while above exit threshold even if below enter threshold.
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.22) == 'RACE'
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.10) == 'SAFE'


def test_scheduler_obeys_min_dwell_before_switch():
    scheduler = FiraProfileScheduler(
        enabled=True,
        default_profile='SAFE',
        min_dwell_frames=3,
        race_enter_severity=0.30,
    )

    # First frames should not switch because dwell has not elapsed.
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.40) == 'SAFE'
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.40) == 'SAFE'
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.40) == 'SAFE'
    # Next frame can switch.
    assert scheduler.run(lane_confidence=1.0, obstacle_severity=0.40) == 'RACE'


def test_scheduler_forces_safe_when_lane_confidence_low():
    scheduler = FiraProfileScheduler(
        enabled=True,
        default_profile='RACE',
        min_dwell_frames=0,
        low_lane_conf_to_safe=0.25,
        race_enter_severity=0.30,
    )

    assert scheduler.run(lane_confidence=0.9, obstacle_severity=0.35) == 'RACE'
    assert scheduler.run(lane_confidence=0.10, obstacle_severity=0.80) == 'SAFE'
