#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run repeatable FIRA readiness battery.')
    p.add_argument('--python-bin', default=sys.executable, help='Python executable to use.')
    p.add_argument('--skip-tests', action='store_true')
    return p


def _run(cmd):
    print('>> ' + ' '.join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _check_competition_config(python_bin: str) -> int:
    checks = {
        'urban': [
            'cfg.WEB_CONTROL_ENABLED is False',
            'cfg.WEBRTC_ENABLED is False',
            'cfg.USE_FPV is False',
            'cfg.USE_NETWORKED_JS is False',
            'cfg.FIRA_MODULAR is True',
            'cfg.FIRA_CHALLENGE_ENABLED is False',
            'cfg.FIRA_CHECKPOINT_DETECTOR_ENABLED is False',
            'cfg.FIRA_REQUIRE_ZEBRA is True',
        ],
        'race': [
            'cfg.WEB_CONTROL_ENABLED is False',
            'cfg.WEBRTC_ENABLED is False',
            'cfg.USE_FPV is False',
            'cfg.USE_NETWORKED_JS is False',
            'cfg.FIRA_MODULAR is False',
            'cfg.FIRA_CHALLENGE_ENABLED is True',
            'cfg.FIRA_PROFILE_SCHEDULER_ENABLED is True',
            'cfg.FIRA_CHECKPOINT_DETECTOR_ENABLED is True',
            'cfg.FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED is True',
        ],
    }

    for mode, assertions in checks.items():
        snippet = [
            'from donkeycar.templates import cfg_competition as cfg',
            f"assert cfg.FIRA_COMPETITION_MODE == '{mode}'",
        ] + [f'assert {a}' for a in assertions]
        cmd = [python_bin, '-c', '; '.join(snippet)]
        env = dict(os.environ)
        env['FIRA_COMPETITION_MODE'] = mode
        print('>> ' + ' '.join(cmd) + f'  # mode={mode}')
        code = subprocess.run(cmd, check=False, env=env).returncode
        if code != 0:
            return code
    return 0


def main() -> int:
    args = _parser().parse_args()

    print('FIRA readiness battery:')
    print('1) Competition config restrictions check')
    print('2) Focused unit/integration tests')
    print('3) Offline official scoring replay smoke')

    code = _check_competition_config(args.python_bin)
    if code != 0:
        return code

    if not args.skip_tests:
        code = _run([
            args.python_bin,
            '-m',
            'pytest',
            '-q',
            'donkeycar/tests/test_fira_competition_metrics.py',
            'donkeycar/tests/test_fira_competition_reporter.py',
            'donkeycar/tests/test_fira_challenge.py',
            'donkeycar/tests/test_fira_multi_obstacle.py',
            'donkeycar/tests/test_fira_predictive_obstacles.py',
        ])
        if code != 0:
            return code

    code = _run([
        args.python_bin,
        'scripts/fira_official_scoring_replay.py',
        '--reports-glob',
        'data/**/fira_competition_report_*.json',
    ])
    # Do not fail battery if there are no reports yet.
    if code not in (0, 1):
        return code

    print('Battery finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
