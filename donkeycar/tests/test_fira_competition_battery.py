import importlib.util
from pathlib import Path


def _load_battery_module():
    script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'fira_competition_battery.py'
    spec = importlib.util.spec_from_file_location('fira_competition_battery_test', script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_battery_skips_replay_and_summary_when_no_reports(monkeypatch):
    mod = _load_battery_module()

    calls = []
    monkeypatch.setattr(mod, '_check_competition_config', lambda _python: 0)
    monkeypatch.setattr(mod.glob, 'glob', lambda _pattern, recursive=True: [])
    monkeypatch.setattr(mod, '_run', lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(mod.sys, 'argv', ['fira_competition_battery.py', '--skip-tests'])

    code = mod.main()

    assert code == 0
    assert calls == []


def test_battery_runs_replay_when_reports_exist(monkeypatch):
    mod = _load_battery_module()

    calls = []
    monkeypatch.setattr(mod, '_check_competition_config', lambda _python: 0)
    monkeypatch.setattr(mod.glob, 'glob', lambda _pattern, recursive=True: ['/tmp/fira_competition_report_1.json'])
    monkeypatch.setattr(mod, '_run', lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(mod.sys, 'argv', ['fira_competition_battery.py', '--skip-tests', '--skip-summary'])

    code = mod.main()

    assert code == 0
    assert len(calls) == 1
    assert 'fira_official_scoring_replay.py' in calls[0][1]


def test_battery_accepts_summary_code_2(monkeypatch):
    mod = _load_battery_module()

    calls = []
    returns = [0, 2]

    def _fake_run(cmd):
        calls.append(cmd)
        return returns.pop(0)

    monkeypatch.setattr(mod, '_check_competition_config', lambda _python: 0)
    monkeypatch.setattr(mod.glob, 'glob', lambda _pattern, recursive=True: ['/tmp/fira_competition_report_1.json'])
    monkeypatch.setattr(mod, '_run', _fake_run)
    monkeypatch.setattr(mod.sys, 'argv', ['fira_competition_battery.py', '--skip-tests'])

    code = mod.main()

    assert code == 0
    assert len(calls) == 2
    assert 'fira_official_scoring_replay.py' in calls[0][1]
    assert 'fira_competition_report_summary.py' in calls[1][1]