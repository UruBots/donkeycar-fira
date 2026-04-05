#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run repeatable FIRA readiness battery.')
    p.add_argument('--python-bin', default=sys.executable, help='Python executable to use.')
    p.add_argument('--skip-tests', action='store_true')
    p.add_argument('--skip-summary', action='store_true')
    p.add_argument('--summary-target-ready-ratio', type=float, default=0.65)
    p.add_argument('--summary-min-active-frames', type=int, default=60)
    p.add_argument('--summary-output-json', default='')
    p.add_argument('--summary-output-md', default='')
    return p


def _run(cmd):
    print('>> ' + ' '.join(cmd))
    env = dict(os.environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ROOT_DIR + (os.pathsep + existing if existing else '')
    return subprocess.run(cmd, check=False, cwd=ROOT_DIR, env=env).returncode


def _check_competition_config(python_bin: str) -> int:
    _ = python_bin

    cfg_path = os.path.join(ROOT_DIR, 'donkeycar', 'templates', 'cfg_competition.py')
    if not os.path.exists(cfg_path):
        print(f'Missing competition config file: {cfg_path}')
        return 2

    with open(cfg_path, 'r', encoding='utf-8') as f:
        text = f.read()

    def _has_all(snippets, scope_name: str) -> bool:
        missing = [s for s in snippets if s not in text]
        if missing:
            print(f'Competition config static check failed in {scope_name}. Missing:')
            for token in missing:
                print(f'  - {token}')
            return False
        return True

    global_tokens = [
        "FIRA_COMPETITION_MODE = os.getenv('FIRA_COMPETITION_MODE', 'urban').strip().lower()",
        'WEB_CONTROL_ENABLED = False',
        'WEBRTC_ENABLED = False',
        'USE_FPV = False',
        'USE_NETWORKED_JS = False',
        'FIRA_COMPETITION_REPORT_ENABLED = True',
    ]
    urban_tokens = [
        "if FIRA_COMPETITION_MODE == 'urban':",
        'FIRA_MODULAR = True',
        'FIRA_CHALLENGE_ENABLED = False',
        'FIRA_CHECKPOINT_DETECTOR_ENABLED = False',
        'FIRA_REQUIRE_ZEBRA = True',
    ]
    race_tokens = [
        "elif FIRA_COMPETITION_MODE == 'race':",
        'FIRA_MODULAR = False',
        'FIRA_CHALLENGE_ENABLED = True',
        'FIRA_PROFILE_SCHEDULER_ENABLED = True',
        'FIRA_CHECKPOINT_DETECTOR_ENABLED = True',
        'FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED = True',
    ]

    if not _has_all(global_tokens, 'global profile'):
        return 3
    if not _has_all(urban_tokens, 'urban profile'):
        return 4
    if not _has_all(race_tokens, 'race profile'):
        return 5

    print('Competition config static profile checks passed.')
    return 0


def main() -> int:
    args = _parser().parse_args()

    print('FIRA readiness battery:')
    print('1) Competition config restrictions check')
    print('2) Focused unit/integration tests')
    print('3) Offline official scoring replay smoke')
    print('4) Session report summary and threshold recommendation')

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

    reports_glob = os.path.join(ROOT_DIR, 'data', '**', 'fira_competition_report_*.json')
    report_paths = sorted(glob.glob(reports_glob, recursive=True))
    if not report_paths:
        print(f'No reports found ({reports_glob}); skipping replay and summary steps.')
        print('Battery finished.')
        return 0

    code = _run([
        args.python_bin,
        os.path.join(ROOT_DIR, 'scripts', 'fira_official_scoring_replay.py'),
        '--reports-glob',
        reports_glob,
    ])
    if code != 0:
        return code

    if not args.skip_summary:
        summary_cmd = [
            args.python_bin,
            os.path.join(ROOT_DIR, 'scripts', 'fira_competition_report_summary.py'),
            '--reports-glob',
            reports_glob,
            '--target-ready-ratio',
            str(args.summary_target_ready_ratio),
            '--min-active-frames',
            str(args.summary_min_active_frames),
        ]
        if args.summary_output_json:
            summary_cmd += ['--output-json', args.summary_output_json]
        if args.summary_output_md:
            summary_cmd += ['--output-md', args.summary_output_md]

        code = _run(summary_cmd)
        # Summary returns 2 when all reports are filtered by min_active_frames.
        if code not in (0, 2):
            return code

    print('Battery finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
