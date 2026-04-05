#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from statistics import mean

from donkeycar.utilities.fira_official_scoring import race_score, total_score, urban_score


def _float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Replay official FIRA scoring from session reports.')
    p.add_argument('--reports-glob', default='data/**/fira_competition_report_*.json')
    p.add_argument('--t-stage-race', type=float, default=120.0)
    p.add_argument('--t-stage-urban', type=float, default=120.0)
    p.add_argument('--total-checkpoints-race', type=int, default=12)
    p.add_argument('--total-checkpoints-urban', type=int, default=12)
    p.add_argument('--autonomy-coeff', type=float, default=1.0)
    p.add_argument('--sign-method', choices=['apriltag', 'vision'], default='apriltag')
    p.add_argument('--print-each', action='store_true')
    return p


def _sign_coeff(method: str) -> float:
    return 1.0 if method == 'apriltag' else 1.3


def main() -> int:
    args = _parser().parse_args()
    paths = sorted(glob.glob(args.reports_glob))
    if not paths:
        print(f'No report files found for glob: {args.reports_glob}')
        return 1

    ks = _sign_coeff(args.sign_method)
    race_scores = []
    urban_scores = []
    total_scores = []

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        metrics = payload.get('metrics', {}) if isinstance(payload, dict) else {}
        dur = _float(payload.get('session_duration_sec', 0.0), 0.0)
        cp = _int(metrics.get('checkpoint_count', 0), 0)

        # Optional explicit penalties, if the report includes them.
        skipped = _int(metrics.get('race_skipped_checkpoints', 0), 0)
        parts_fell = _int(metrics.get('race_parts_fell', 0), 0)
        no_stop = _int(metrics.get('urban_no_stop_count', 0), 0)
        bad_turn = _int(metrics.get('urban_incorrect_turn_count', 0), 0)
        bad_lane_change = _int(metrics.get('urban_incorrect_lane_change_count', 0), 0)

        sar = race_score(
            t_stage=args.t_stage_race,
            t_total=dur,
            checkpoints_passed=cp,
            total_checkpoints=args.total_checkpoints_race,
            skipped_checkpoints=skipped,
            parts_fell=parts_fell,
        )
        saud = urban_score(
            t_stage=args.t_stage_urban,
            t_total=dur,
            checkpoints_passed=cp,
            total_checkpoints=args.total_checkpoints_urban,
            no_stop_count=no_stop,
            incorrect_turn_count=bad_turn,
            incorrect_lane_change_count=bad_lane_change,
            sign_method_coeff=ks,
        )
        st = total_score(sar, saud, autonomy_coeff=args.autonomy_coeff)

        race_scores.append(sar)
        urban_scores.append(saud)
        total_scores.append(st)

        if args.print_each:
            print(f'{path}: SAR={sar:.3f} SAUD={saud:.3f} ST={st:.3f}')

    print(f'reports={len(paths)}')
    print(f'SAR max={max(race_scores):.3f} avg={mean(race_scores):.3f}')
    print(f'SAUD max={max(urban_scores):.3f} avg={mean(urban_scores):.3f}')
    print(f'ST max={max(total_scores):.3f} avg={mean(total_scores):.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
