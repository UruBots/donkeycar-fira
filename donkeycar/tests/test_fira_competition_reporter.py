import json
from pathlib import Path

from donkeycar.parts.fira_competition_reporter import FiraCompetitionSessionReporter


def test_reporter_writes_json_when_recording_stops(tmp_path: Path):
    reporter = FiraCompetitionSessionReporter(
        output_dir=str(tmp_path),
        enabled=True,
        min_active_frames=3,
    )

    # Initial state
    reporter.run(recording=False)

    # Start recording session
    reporter.run(recording=True)

    # Stop recording with enough activity -> should write report
    reporter.run(
        recording=False,
        user_mode='local',
        right_lane_score=0.82,
        checkpoint_count=5,
        checkpoint_progress=0.50,
        lane_violations=2,
        active_frames=10,
        compliance_score=0.76,
        compliance_ready=True,
    )

    reports = sorted(tmp_path.glob('fira_competition_report_*.json'))
    assert len(reports) == 1

    payload = json.loads(reports[0].read_text(encoding='utf-8'))
    assert payload['metrics']['right_lane_score'] == 0.82
    assert payload['metrics']['checkpoint_count'] == 5
    assert payload['metrics']['checkpoint_progress'] == 0.5
    assert payload['metrics']['lane_violations'] == 2
    assert payload['metrics']['active_frames'] == 10
    assert payload['metrics']['compliance_score'] == 0.76
    assert payload['metrics']['compliance_ready'] is True


def test_reporter_skips_when_active_frames_below_threshold(tmp_path: Path):
    reporter = FiraCompetitionSessionReporter(
        output_dir=str(tmp_path),
        enabled=True,
        min_active_frames=20,
    )

    reporter.run(recording=False)
    reporter.run(recording=True)
    reporter.run(
        recording=False,
        right_lane_score=0.6,
        checkpoint_count=2,
        checkpoint_progress=0.2,
        lane_violations=1,
        active_frames=8,
        compliance_score=0.4,
        compliance_ready=False,
    )

    reports = list(tmp_path.glob('fira_competition_report_*.json'))
    assert reports == []
