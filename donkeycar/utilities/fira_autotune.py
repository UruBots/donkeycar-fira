from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SessionMetrics:
    right_lane_score: float
    checkpoint_progress: float
    lane_violations: int
    active_frames: int
    compliance_score: float
    official_race_score: float = 0.0
    official_urban_score: float = 0.0


PARAM_BOUNDS = {
    'FIRA_CHALLENGE_ENGAGE_SEVERITY': (0.05, 0.35),
    'FIRA_CHALLENGE_CLEAR_FRAMES': (1, 6),
    'FIRA_CHALLENGE_RECOVER_FRAMES': (2, 12),
    'FIRA_CHALLENGE_SMOOTHING_ALPHA': (0.15, 0.85),
    'FIRA_CHALLENGE_MAX_ABS_CORRECTION': (0.20, 0.80),
    'FIRA_CHALLENGE_PREDICTIVE_BLEND': (0.25, 0.85),
}

DEFAULT_BASELINE = {
    'FIRA_CHALLENGE_ENGAGE_SEVERITY': 0.12,
    'FIRA_CHALLENGE_CLEAR_FRAMES': 2,
    'FIRA_CHALLENGE_RECOVER_FRAMES': 6,
    'FIRA_CHALLENGE_SMOOTHING_ALPHA': 0.35,
    'FIRA_CHALLENGE_MAX_ABS_CORRECTION': 0.70,
    'FIRA_CHALLENGE_PREDICTIVE_BLEND': 0.55,
}


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def load_reports(reports_glob: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(reports_glob)):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def parse_sessions(payloads: Sequence[dict]) -> List[SessionMetrics]:
    sessions: List[SessionMetrics] = []
    for row in payloads:
        metrics = row.get('metrics', {}) if isinstance(row, dict) else {}
        try:
            session = SessionMetrics(
                right_lane_score=float(metrics.get('right_lane_score', 0.0)),
                checkpoint_progress=float(metrics.get('checkpoint_progress', 0.0)),
                lane_violations=int(metrics.get('lane_violations', 0)),
                active_frames=int(metrics.get('active_frames', 0)),
                compliance_score=float(metrics.get('compliance_score', 0.0)),
                official_race_score=float(metrics.get('official_race_score', 0.0)),
                official_urban_score=float(metrics.get('official_urban_score', 0.0)),
            )
        except (TypeError, ValueError):
            continue
        if session.active_frames > 0:
            sessions.append(session)
    return sessions


def split_sessions(
    sessions: Sequence[SessionMetrics],
    train_ratio: float,
    seed: int,
) -> Tuple[List[SessionMetrics], List[SessionMetrics]]:
    if not sessions:
        return [], []
    idx = list(range(len(sessions)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    cut = max(1, min(len(idx) - 1, int(round(_clip(train_ratio, 0.1, 0.9) * len(idx))))) if len(idx) > 1 else 1
    train = [sessions[i] for i in idx[:cut]]
    valid = [sessions[i] for i in idx[cut:]]
    if not valid:
        valid = train[:]
    return train, valid


def apply_guardrails(candidate: Dict[str, float]) -> Dict[str, float]:
    tuned: Dict[str, float] = {}
    for key, bounds in PARAM_BOUNDS.items():
        low, high = bounds
        value = candidate.get(key, DEFAULT_BASELINE[key])
        if isinstance(low, int) and isinstance(high, int):
            tuned[key] = int(round(_clip(float(value), float(low), float(high))))
        else:
            tuned[key] = float(_clip(float(value), float(low), float(high)))

    # Stability guardrails.
    tuned['FIRA_CHALLENGE_RECOVER_FRAMES'] = max(
        int(tuned['FIRA_CHALLENGE_RECOVER_FRAMES']),
        int(tuned['FIRA_CHALLENGE_CLEAR_FRAMES']) + 1,
    )
    tuned['FIRA_CHALLENGE_MAX_ABS_CORRECTION'] = min(
        float(tuned['FIRA_CHALLENGE_MAX_ABS_CORRECTION']),
        0.75,
    )
    return tuned


def _session_base_score(session: SessionMetrics, target_active_frames: int) -> float:
    survival = _clip(session.active_frames / max(1.0, float(target_active_frames)), 0.0, 1.0)
    violations_rate = session.lane_violations / max(1.0, float(session.active_frames))
    safety = 1.0 - _clip(violations_rate * 16.0, 0.0, 1.0)
    heuristics = (
        0.30 * _clip(session.checkpoint_progress, 0.0, 1.0)
        + 0.25 * _clip(session.compliance_score, 0.0, 1.0)
        + 0.20 * _clip(session.right_lane_score, 0.0, 1.0)
        + 0.15 * survival
        + 0.10 * safety
    )
    if session.official_race_score > 0.0 or session.official_urban_score > 0.0:
        # Normalize official scores to [0, 1] range for stable blending.
        official_norm = _clip((session.official_race_score + session.official_urban_score) / 200.0, 0.0, 1.0)
        return 0.55 * official_norm + 0.45 * heuristics
    return heuristics


def _aggression(candidate: Dict[str, float]) -> float:
    engage = candidate['FIRA_CHALLENGE_ENGAGE_SEVERITY']
    clear_f = candidate['FIRA_CHALLENGE_CLEAR_FRAMES']
    recover_f = candidate['FIRA_CHALLENGE_RECOVER_FRAMES']
    corr = candidate['FIRA_CHALLENGE_MAX_ABS_CORRECTION']
    smooth = candidate['FIRA_CHALLENGE_SMOOTHING_ALPHA']

    engage_term = 1.0 - _clip((engage - 0.05) / (0.35 - 0.05), 0.0, 1.0)
    clear_term = 1.0 - _clip((clear_f - 1.0) / 5.0, 0.0, 1.0)
    recover_term = 1.0 - _clip((recover_f - 2.0) / 10.0, 0.0, 1.0)
    corr_term = _clip((corr - 0.2) / 0.6, 0.0, 1.0)
    smooth_term = 1.0 - _clip((smooth - 0.15) / 0.7, 0.0, 1.0)

    return _clip(
        0.24 * engage_term + 0.16 * clear_term + 0.16 * recover_term + 0.26 * corr_term + 0.18 * smooth_term,
        0.0,
        1.0,
    )


def score_candidate(
    candidate: Dict[str, float],
    sessions: Sequence[SessionMetrics],
    target_active_frames: int,
) -> float:
    if not sessions:
        return 0.0

    aggression = _aggression(candidate)
    predictive = _clip(candidate['FIRA_CHALLENGE_PREDICTIVE_BLEND'], 0.0, 1.0)

    total = 0.0
    for s in sessions:
        base = _session_base_score(s, target_active_frames)
        violations_rate = s.lane_violations / max(1.0, float(s.active_frames))
        challenge_density = _clip((1.0 - s.right_lane_score) + (1.0 - s.checkpoint_progress), 0.0, 1.0)

        progress_bonus = aggression * 0.20 * challenge_density
        safety_penalty = aggression * 0.30 * _clip(violations_rate * 20.0, 0.0, 1.0)
        smoothness_penalty = aggression * (1.0 - predictive) * 0.10
        predictive_bonus = predictive * 0.08 * _clip(s.checkpoint_progress + s.compliance_score, 0.0, 1.0)

        total += _clip(base + progress_bonus + predictive_bonus - safety_penalty - smoothness_penalty, 0.0, 1.0)

    return total / float(len(sessions))


def _sample_candidate(rng: random.Random, baseline: Dict[str, float]) -> Dict[str, float]:
    c = dict(baseline)
    for key, (low, high) in PARAM_BOUNDS.items():
        if isinstance(low, int) and isinstance(high, int):
            c[key] = rng.randint(int(low), int(high))
        else:
            c[key] = rng.uniform(float(low), float(high))
    return apply_guardrails(c)


def run_autotune(
    sessions: Sequence[SessionMetrics],
    baseline: Dict[str, float] | None = None,
    trials: int = 80,
    seed: int = 42,
    train_ratio: float = 0.7,
    target_active_frames: int = 120,
) -> dict:
    baseline_params = apply_guardrails(dict(DEFAULT_BASELINE if baseline is None else baseline))
    train, valid = split_sessions(sessions, train_ratio=train_ratio, seed=seed)
    eval_valid = valid if valid else train

    baseline_train = score_candidate(baseline_params, train, target_active_frames)
    baseline_valid = score_candidate(baseline_params, eval_valid, target_active_frames)

    best = dict(baseline_params)
    best_train = baseline_train
    best_valid = baseline_valid

    rng = random.Random(seed)
    trial_rows = []
    for idx in range(max(1, int(trials))):
        candidate = _sample_candidate(rng, baseline_params)
        train_score = score_candidate(candidate, train, target_active_frames)
        valid_score = score_candidate(candidate, eval_valid, target_active_frames)
        unstable = (
            candidate['FIRA_CHALLENGE_MAX_ABS_CORRECTION'] > 0.73
            and candidate['FIRA_CHALLENGE_SMOOTHING_ALPHA'] < 0.22
        )
        rejected = unstable or (valid_score + 1e-6 < baseline_valid - 0.03)
        trial_rows.append(
            {
                'trial': idx,
                'rejected': rejected,
                'train_score': train_score,
                'valid_score': valid_score,
                'params': candidate,
            }
        )
        if not rejected and (train_score > best_train + 1e-9):
            best = candidate
            best_train = train_score
            best_valid = valid_score

    # Enforce non-regression on validation split.
    if best_valid + 1e-9 < baseline_valid:
        best = dict(baseline_params)
        best_train = baseline_train
        best_valid = baseline_valid

    return {
        'seed': seed,
        'trials': int(trials),
        'sessions_total': len(sessions),
        'sessions_train': len(train),
        'sessions_valid': len(eval_valid),
        'target_active_frames': int(target_active_frames),
        'baseline': {
            'train_score': baseline_train,
            'valid_score': baseline_valid,
            'params': baseline_params,
        },
        'best': {
            'train_score': best_train,
            'valid_score': best_valid,
            'params': best,
        },
        'improvement': {
            'train_delta': best_train - baseline_train,
            'valid_delta': best_valid - baseline_valid,
        },
        'trials_top5': sorted(
            [r for r in trial_rows if not r['rejected']],
            key=lambda r: (r['train_score'], r['valid_score']),
            reverse=True,
        )[:5],
    }


def format_overrides(params: Dict[str, float]) -> str:
    lines = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            lines.append(f"{key} = {value:.4f}")
        else:
            lines.append(f"{key} = {value}")
    return '\n'.join(lines) + '\n'


def summarize_report(report: dict) -> str:
    base = report['baseline']
    best = report['best']
    imp = report['improvement']
    return (
        f"sessions={report['sessions_total']} train={report['sessions_train']} valid={report['sessions_valid']}\n"
        f"baseline(train={base['train_score']:.4f}, valid={base['valid_score']:.4f})\n"
        f"best(train={best['train_score']:.4f}, valid={best['valid_score']:.4f})\n"
        f"delta(train={imp['train_delta']:.4f}, valid={imp['valid_delta']:.4f})"
    )
