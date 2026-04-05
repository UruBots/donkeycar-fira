from __future__ import annotations


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def race_penalty_time(
    t_stage: float,
    total_checkpoints: int,
    skipped_checkpoints: int,
    parts_fell: int,
) -> float:
    t_stage = max(1e-6, float(t_stage))
    total_checkpoints = max(1, int(total_checkpoints))
    skipped_checkpoints = max(0, int(skipped_checkpoints))
    parts_fell = max(0, int(parts_fell))

    unit = t_stage / float(total_checkpoints)
    return (0.5 * unit * skipped_checkpoints) + (0.2 * unit * parts_fell)


def race_score(
    t_stage: float,
    t_total: float,
    checkpoints_passed: int,
    total_checkpoints: int,
    skipped_checkpoints: int = 0,
    parts_fell: int = 0,
    extra_penalty_points: float = 0.0,
) -> float:
    """Race score per FIRA formula section 5.4 with time penalties from 5.3."""
    t_stage = max(1e-6, float(t_stage))
    t_total = max(0.0, float(t_total))
    cp = max(0, int(checkpoints_passed))
    total_cp = max(1, int(total_checkpoints))
    p_points = max(0.0, float(extra_penalty_points))

    penalty_time = race_penalty_time(t_stage, total_cp, skipped_checkpoints, parts_fell)
    t_eff = t_total + penalty_time
    time_factor = _clip(1.0 - (t_eff / t_stage), 0.0, 1.0)
    cp_factor = _clip(cp / float(total_cp), 0.0, 1.0)

    score = (100.0 * time_factor * cp_factor) - p_points
    return max(0.0, score)


def urban_score(
    t_stage: float,
    t_total: float,
    checkpoints_passed: int,
    total_checkpoints: int,
    no_stop_count: int = 0,
    incorrect_turn_count: int = 0,
    incorrect_lane_change_count: int = 0,
    sign_method_coeff: float = 1.0,
) -> float:
    """Urban score per FIRA section 6.6 + penalties from 6.5 + Ks from 6.4."""
    t_stage = max(1e-6, float(t_stage))
    t_total = max(0.0, float(t_total))
    cp = max(0, int(checkpoints_passed))
    total_cp = max(1, int(total_checkpoints))

    p = (15 * max(0, int(no_stop_count))) + (25 * max(0, int(incorrect_turn_count))) + (20 * max(0, int(incorrect_lane_change_count)))
    ks = max(0.0, float(sign_method_coeff))

    time_factor = _clip(1.0 - (t_total / t_stage), 0.0, 1.0)
    cp_factor = _clip(cp / float(total_cp), 0.0, 1.0)

    score = ((100.0 * time_factor * cp_factor) - float(p)) * ks
    return max(0.0, score)


def total_score(sa_race: float, sa_urban: float, autonomy_coeff: float = 1.0) -> float:
    return max(0.0, (max(0.0, float(sa_race)) + max(0.0, float(sa_urban))) * max(0.0, float(autonomy_coeff)))
