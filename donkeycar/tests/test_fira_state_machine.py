import pytest

from donkeycar.parts.fira_state_machine import FiraDriveStateMachine


def test_startup_holds_initially(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr('donkeycar.parts.fira_state_machine.time.time', _time)

    machine = FiraDriveStateMachine(startup_hold_seconds=2.0)
    state, cap = machine.run()

    assert state == 'startup'
    assert cap == 0.0


def test_detector_down_forces_recovery(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr('donkeycar.parts.fira_state_machine.time.time', _time)

    machine = FiraDriveStateMachine(startup_hold_seconds=0.0, recovery_hold_seconds=0.0)
    state, cap = machine.run(
        yolo_state='idle',
        yolo_health={'detector_ready': False, 'detector_backend': 'unavailable'},
        lane_confidence=1.0,
        obstacle_severity=0.0,
    )

    assert state == 'recovery'
    assert cap == pytest.approx(0.05)


def test_low_lane_confidence_uses_normal_recovery_cap(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr('donkeycar.parts.fira_state_machine.time.time', _time)

    machine = FiraDriveStateMachine(startup_hold_seconds=0.0, recovery_hold_seconds=0.0)
    state, cap = machine.run(
        yolo_state='idle',
        yolo_health={'detector_ready': True, 'detector_backend': 'ultralytics'},
        lane_confidence=0.1,
        obstacle_severity=0.0,
    )

    assert state == 'recovery'
    assert cap == pytest.approx(0.15)


def test_crosswalk_state_gets_slow_cap(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr('donkeycar.parts.fira_state_machine.time.time', _time)

    machine = FiraDriveStateMachine(startup_hold_seconds=0.0, zebra_hold_seconds=0.0)
    state, cap = machine.run(
        yolo_state='wait-at-crosswalk',
        yolo_health={'detector_ready': True, 'detector_backend': 'ultralytics'},
        lane_confidence=0.9,
        obstacle_severity=0.1,
    )

    assert state == 'zebra_slow'
    assert cap == pytest.approx(0.25)


def test_nominal_drive_becomes_race_fast(monkeypatch):
    fake_time = {'value': 10.0}

    def _time():
        return fake_time['value']

    monkeypatch.setattr('donkeycar.parts.fira_state_machine.time.time', _time)

    machine = FiraDriveStateMachine(startup_hold_seconds=0.0)
    state, cap = machine.run(
        yolo_state='idle',
        yolo_health={'detector_ready': True, 'detector_backend': 'ultralytics'},
        lane_confidence=0.95,
        obstacle_severity=0.0,
    )

    assert state == 'race_fast'
    assert cap == pytest.approx(1.0)
