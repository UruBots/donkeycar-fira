import json
from pathlib import Path

from donkeycar.utilities.fira_autotune import (
    DEFAULT_BASELINE,
    apply_guardrails,
    format_overrides,
    load_reports,
    parse_sessions,
    run_autotune,
)


def _report(right_lane, progress, violations, frames, compliance):
    return {
        'metrics': {
            'right_lane_score': right_lane,
            'checkpoint_progress': progress,
            'lane_violations': violations,
            'active_frames': frames,
            'compliance_score': compliance,
        }
    }


def test_parse_sessions_filters_invalid_rows():
    payloads = [
        _report(0.7, 0.4, 3, 150, 0.6),
        {'metrics': {'active_frames': 0}},
        {'metrics': {'right_lane_score': 'bad'}},
    ]
    sessions = parse_sessions(payloads)
    assert len(sessions) == 1
    assert sessions[0].active_frames == 150


def test_guardrails_clip_and_enforce_stability():
    candidate = {
        'FIRA_CHALLENGE_ENGAGE_SEVERITY': 0.01,
        'FIRA_CHALLENGE_CLEAR_FRAMES': 6,
        'FIRA_CHALLENGE_RECOVER_FRAMES': 2,
        'FIRA_CHALLENGE_SMOOTHING_ALPHA': 0.05,
        'FIRA_CHALLENGE_MAX_ABS_CORRECTION': 1.3,
        'FIRA_CHALLENGE_PREDICTIVE_BLEND': 0.99,
    }
    tuned = apply_guardrails(candidate)
    assert 0.05 <= tuned['FIRA_CHALLENGE_ENGAGE_SEVERITY'] <= 0.35
    assert tuned['FIRA_CHALLENGE_RECOVER_FRAMES'] >= tuned['FIRA_CHALLENGE_CLEAR_FRAMES'] + 1
    assert tuned['FIRA_CHALLENGE_MAX_ABS_CORRECTION'] <= 0.75


def test_autotune_non_regression_on_validation_split():
    sessions = [
        parse_sessions([_report(0.55, 0.25, 10, 140, 0.50)])[0],
        parse_sessions([_report(0.60, 0.35, 8, 160, 0.58)])[0],
        parse_sessions([_report(0.70, 0.45, 4, 190, 0.66)])[0],
        parse_sessions([_report(0.75, 0.55, 2, 210, 0.72)])[0],
    ]

    report = run_autotune(
        sessions=sessions,
        baseline=dict(DEFAULT_BASELINE),
        trials=40,
        seed=7,
        train_ratio=0.5,
        target_active_frames=120,
    )

    assert report['best']['valid_score'] >= report['baseline']['valid_score'] - 1e-9
    assert report['sessions_total'] == 4
    assert len(report['trials_top5']) <= 5


def test_load_reports_and_format_overrides(tmp_path: Path):
    p = tmp_path / 'fira_competition_report_1.json'
    p.write_text(json.dumps(_report(0.7, 0.4, 3, 150, 0.6)), encoding='utf-8')

    rows = load_reports(str(tmp_path / 'fira_competition_report_*.json'))
    assert len(rows) == 1

    text = format_overrides({'FIRA_CHALLENGE_CLEAR_FRAMES': 3, 'FIRA_CHALLENGE_SMOOTHING_ALPHA': 0.4})
    assert 'FIRA_CHALLENGE_CLEAR_FRAMES = 3' in text
    assert 'FIRA_CHALLENGE_SMOOTHING_ALPHA = 0.4000' in text
