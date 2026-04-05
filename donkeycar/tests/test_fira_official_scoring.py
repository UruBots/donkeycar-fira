from donkeycar.utilities.fira_official_scoring import race_score, total_score, urban_score


def test_race_score_decreases_with_penalties():
    clean = race_score(
        t_stage=120.0,
        t_total=80.0,
        checkpoints_passed=10,
        total_checkpoints=12,
    )
    penalized = race_score(
        t_stage=120.0,
        t_total=80.0,
        checkpoints_passed=10,
        total_checkpoints=12,
        skipped_checkpoints=2,
        parts_fell=1,
    )
    assert penalized < clean


def test_urban_score_applies_ks_and_penalties():
    apriltag = urban_score(
        t_stage=120.0,
        t_total=70.0,
        checkpoints_passed=9,
        total_checkpoints=12,
        sign_method_coeff=1.0,
    )
    vision = urban_score(
        t_stage=120.0,
        t_total=70.0,
        checkpoints_passed=9,
        total_checkpoints=12,
        sign_method_coeff=1.3,
    )
    bad = urban_score(
        t_stage=120.0,
        t_total=70.0,
        checkpoints_passed=9,
        total_checkpoints=12,
        no_stop_count=2,
        incorrect_turn_count=1,
        incorrect_lane_change_count=1,
        sign_method_coeff=1.0,
    )
    assert vision > apriltag
    assert bad < apriltag


def test_total_score_combines_both_categories_with_autonomy_coeff():
    st = total_score(40.0, 50.0, autonomy_coeff=1.0)
    st_half = total_score(40.0, 50.0, autonomy_coeff=0.5)
    assert st == 90.0
    assert st_half == 45.0
