import json
import subprocess
import sys
from pathlib import Path


def _write_report(path: Path, compliance: float, ready: bool, active: int, checkpoints: int = 0):
    payload = {
        'session_duration_sec': 90.0,
        'metrics': {
            'right_lane_score': 0.7,
            'checkpoint_count': checkpoints,
            'checkpoint_progress': 0.5,
            'lane_violations': 4,
            'active_frames': active,
            'compliance_score': compliance,
            'compliance_ready': ready,
        },
    }
    path.write_text(json.dumps(payload), encoding='utf-8')


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'scripts' / 'fira_competition_report_summary.py'


def test_summary_script_outputs_recommended_threshold_and_json(tmp_path: Path):
    _write_report(tmp_path / 'fira_competition_report_a.json', 0.58, False, 120, checkpoints=4)
    _write_report(tmp_path / 'fira_competition_report_b.json', 0.74, True, 160, checkpoints=6)
    _write_report(tmp_path / 'fira_competition_report_c.json', 0.82, True, 180, checkpoints=8)

    out_json = tmp_path / 'summary.json'
    cmd = [
        sys.executable,
        str(_script_path()),
        '--reports-glob',
        str(tmp_path / 'fira_competition_report_*.json'),
        '--target-ready-ratio',
        '0.67',
        '--output-json',
        str(out_json),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert 'reports=3 valid=3 skipped=0' in proc.stdout
    assert 'recommended FIRA_COMPETITION_COMPLIANCE_SCORE_THRESHOLD=' in proc.stdout
    assert out_json.exists()

    summary = json.loads(out_json.read_text(encoding='utf-8'))
    threshold = summary['recommendations']['recommended_compliance_score_threshold']
    assert 0.58 <= threshold <= 0.82
    assert summary['reports_found'] == 3
    assert summary['reports_valid'] == 3


def test_summary_script_writes_markdown_output(tmp_path: Path):
    _write_report(tmp_path / 'fira_competition_report_a.json', 0.60, False, 140, checkpoints=4)
    _write_report(tmp_path / 'fira_competition_report_b.json', 0.80, True, 170, checkpoints=7)

    out_md = tmp_path / 'summary.md'
    cmd = [
        sys.executable,
        str(_script_path()),
        '--reports-glob',
        str(tmp_path / 'fira_competition_report_*.json'),
        '--output-md',
        str(out_md),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert out_md.exists()
    md = out_md.read_text(encoding='utf-8')
    assert '# FIRA Competition Session Summary' in md
    assert 'Suggested FIRA_COMPETITION_COMPLIANCE_SCORE_THRESHOLD' in md


def test_summary_script_filters_low_active_reports(tmp_path: Path):
    _write_report(tmp_path / 'fira_competition_report_short.json', 0.90, True, 30)
    _write_report(tmp_path / 'fira_competition_report_good.json', 0.70, True, 140)

    cmd = [
        sys.executable,
        str(_script_path()),
        '--reports-glob',
        str(tmp_path / 'fira_competition_report_*.json'),
        '--min-active-frames',
        '100',
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert 'reports=2 valid=1 skipped=1' in proc.stdout


def test_summary_script_returns_1_when_no_reports(tmp_path: Path):
    cmd = [
        sys.executable,
        str(_script_path()),
        '--reports-glob',
        str(tmp_path / 'fira_competition_report_*.json'),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert proc.returncode == 1
    assert 'No report files found for glob:' in proc.stdout