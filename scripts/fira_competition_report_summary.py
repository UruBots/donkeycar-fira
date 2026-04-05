#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from statistics import mean, median


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


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _quantile(values, q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(float(v) for v in values)
    q = _clip(float(q), 0.0, 1.0)
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stats(values):
    if not values:
        return {
            'min': 0.0,
            'max': 0.0,
            'avg': 0.0,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
        }
    return {
        'min': float(min(values)),
        'max': float(max(values)),
        'avg': float(mean(values)),
        'median': float(median(values)),
        'p25': float(_quantile(values, 0.25)),
        'p75': float(_quantile(values, 0.75)),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Summarize FIRA session reports and suggest threshold tuning.')
    p.add_argument('--reports-glob', default='data/**/fira_competition_report_*.json')
    p.add_argument('--min-active-frames', type=int, default=60)
    p.add_argument('--target-ready-ratio', type=float, default=0.65)
    p.add_argument('--output-json', default='')
    p.add_argument('--output-md', default='')
    return p


def _render_markdown(summary: dict) -> str:
    score = summary['metrics']['compliance_score']
    rec = summary['recommendations']
    lines = [
        '# FIRA Competition Session Summary',
        '',
        '## Dataset',
        f"- Reports found: {summary['reports_found']}",
        f"- Reports valid: {summary['reports_valid']}",
        f"- Reports skipped: {summary['reports_skipped']}",
        f"- Min active frames filter: {summary['filters']['min_active_frames']}",
        '',
        '## Compliance Score',
        f"- Avg: {score['avg']:.3f}",
        f"- Median: {score['median']:.3f}",
        f"- P25/P75: {score['p25']:.3f} / {score['p75']:.3f}",
        f"- Min/Max: {score['min']:.3f} / {score['max']:.3f}",
        '',
        '## Recommendation',
        f"- Target ready ratio: {rec['target_ready_ratio']:.3f}",
        f"- Current ready ratio: {rec['current_ready_ratio']:.3f}",
        f"- Predicted ready ratio: {rec['predicted_ready_ratio']:.3f}",
        f"- Suggested {rec['suggested_cfg_key']}: {rec['recommended_compliance_score_threshold']:.3f}",
    ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = _parser().parse_args()

    paths = sorted(glob.glob(args.reports_glob, recursive=True))
    if not paths:
        print(f'No report files found for glob: {args.reports_glob}')
        return 1

    min_active_frames = max(1, int(args.min_active_frames))
    target_ready_ratio = _clip(_float(args.target_ready_ratio, 0.65), 0.05, 0.95)

    rows = []
    skipped = 0
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        metrics = payload.get('metrics', {}) if isinstance(payload, dict) else {}
        active_frames = _int(metrics.get('active_frames', 0), 0)
        if active_frames < min_active_frames:
            skipped += 1
            continue

        lane_violations = _int(metrics.get('lane_violations', 0), 0)
        violation_rate = 0.0
        if active_frames > 0:
            violation_rate = float(lane_violations) / float(active_frames)

        rows.append({
            'path': path,
            'right_lane_score': _clip(_float(metrics.get('right_lane_score', 0.0), 0.0), 0.0, 1.0),
            'checkpoint_progress': _clip(_float(metrics.get('checkpoint_progress', 0.0), 0.0), 0.0, 1.0),
            'checkpoint_count': _int(metrics.get('checkpoint_count', 0), 0),
            'lane_violations': lane_violations,
            'active_frames': active_frames,
            'violation_rate': violation_rate,
            'compliance_score': _clip(_float(metrics.get('compliance_score', 0.0), 0.0), 0.0, 1.0),
            'compliance_ready': bool(metrics.get('compliance_ready', False)),
        })

    if not rows:
        print(
            'No valid reports after filtering '
            f'(glob={args.reports_glob}, min_active_frames={min_active_frames})'
        )
        return 2

    compliance_scores = [r['compliance_score'] for r in rows]
    current_ready_ratio = mean(1.0 if r['compliance_ready'] else 0.0 for r in rows)

    # Pick threshold to roughly match desired ready ratio over recent sessions.
    # Example: target_ready_ratio=0.65 => threshold at q=0.35.
    q = _clip(1.0 - target_ready_ratio, 0.0, 1.0)
    recommended_threshold = _clip(_quantile(compliance_scores, q), 0.50, 0.95)
    predicted_ready_ratio = mean(
        1.0 if r['compliance_score'] >= recommended_threshold else 0.0
        for r in rows
    )

    summary = {
        'reports_found': len(paths),
        'reports_valid': len(rows),
        'reports_skipped': skipped,
        'filters': {
            'min_active_frames': min_active_frames,
        },
        'metrics': {
            'compliance_score': _stats(compliance_scores),
            'right_lane_score': _stats([r['right_lane_score'] for r in rows]),
            'checkpoint_progress': _stats([r['checkpoint_progress'] for r in rows]),
            'violation_rate': _stats([r['violation_rate'] for r in rows]),
            'checkpoint_count': {
                'min': int(min(r['checkpoint_count'] for r in rows)),
                'max': int(max(r['checkpoint_count'] for r in rows)),
                'avg': float(mean(r['checkpoint_count'] for r in rows)),
                'median': float(median(r['checkpoint_count'] for r in rows)),
            },
        },
        'recommendations': {
            'target_ready_ratio': float(target_ready_ratio),
            'current_ready_ratio': float(current_ready_ratio),
            'recommended_compliance_score_threshold': float(recommended_threshold),
            'predicted_ready_ratio': float(predicted_ready_ratio),
            'suggested_cfg_key': 'FIRA_COMPETITION_COMPLIANCE_SCORE_THRESHOLD',
        },
    }

    print(
        f"reports={summary['reports_found']} valid={summary['reports_valid']} "
        f"skipped={summary['reports_skipped']}"
    )
    print(
        'compliance_score '
        f"avg={summary['metrics']['compliance_score']['avg']:.3f} "
        f"p25={summary['metrics']['compliance_score']['p25']:.3f} "
        f"median={summary['metrics']['compliance_score']['median']:.3f} "
        f"p75={summary['metrics']['compliance_score']['p75']:.3f}"
    )
    print(
        f"ready_ratio current={summary['recommendations']['current_ready_ratio']:.3f} "
        f"target={summary['recommendations']['target_ready_ratio']:.3f} "
        f"predicted={summary['recommendations']['predicted_ready_ratio']:.3f}"
    )
    print(
        'recommended '
        f"{summary['recommendations']['suggested_cfg_key']}="
        f"{summary['recommendations']['recommended_compliance_score_threshold']:.3f}"
    )

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f'summary_json={args.output_json}')

    if args.output_md:
        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write(_render_markdown(summary))
        print(f'summary_md={args.output_md}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())