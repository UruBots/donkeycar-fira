import ast
from pathlib import Path


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_function(function_name):
    complete_path = Path(__file__).resolve().parents[1] / 'templates' / 'complete.py'
    source = complete_path.read_text(encoding='utf-8')
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            func_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(func_module)
            namespace = {}
            exec(compile(func_module, str(complete_path), 'exec'), namespace)
            return namespace[function_name]

    raise AssertionError(f'{function_name} not found in complete.py')


def test_safety_signal_estimator_inputs_only_include_enabled_detectors():
    func = _load_function('fira_safety_signal_estimator_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=True,
        FIRA_ENGINE_TF=False,
        FIRA_PROFILE_SCHEDULER_ENABLED=False,
    )

    assert func(cfg) == ['cam/image_array', 'pilot/angle', 'fira/yolo/detector_ready']


def test_safety_signal_estimator_inputs_include_active_profile_when_scheduler_enabled():
    func = _load_function('fira_safety_signal_estimator_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=False,
        FIRA_ENGINE_TF=False,
        FIRA_PROFILE_SCHEDULER_ENABLED=True,
    )

    assert func(cfg) == ['cam/image_array', 'pilot/angle', 'fira/active_profile']


def test_profile_scheduler_inputs_include_signal_context_and_drive_state_when_available():
    func = _load_function('fira_profile_scheduler_inputs')
    cfg = _Cfg(FIRA_SAFETY_SIGNAL_ESTIMATOR=True, FIRA_STATE_MACHINE=True)

    assert func(cfg) == ['fira/lane_confidence', 'fira/obstacle_severity', 'fira/drive/state']


def test_safety_arbiter_inputs_include_only_available_health_and_signals():
    func = _load_function('fira_safety_arbiter_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=True,
        FIRA_ENGINE_TF=False,
        FIRA_SAFETY_SIGNAL_ESTIMATOR=True,
        FIRA_STATE_MACHINE=True,
    )

    assert func(cfg) == [
        'pilot/angle',
        'pilot/throttle',
        'cam/image_array',
        'fira/yolo/health',
        'fira/lane_angle',
        'fira/lane_confidence',
        'fira/obstacle_steering_correction',
        'fira/obstacle_severity',
        'fira/drive/state',
        'fira/drive/state_throttle_cap',
    ]


def test_safety_arbiter_inputs_minimal_without_fira_engines():
    func = _load_function('fira_safety_arbiter_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=False,
        FIRA_ENGINE_TF=False,
        FIRA_SAFETY_SIGNAL_ESTIMATOR=False,
    )

    assert func(cfg) == ['pilot/angle', 'pilot/throttle', 'cam/image_array']


def test_drive_state_machine_inputs_include_engine_state_and_signal_context():
    func = _load_function('fira_drive_state_machine_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=True,
        FIRA_ENGINE_TF=True,
        FIRA_SAFETY_SIGNAL_ESTIMATOR=True,
    )

    assert func(cfg) == [
        'fira/yolo/state',
        'fira/yolo/health',
        'fira/tf/state',
        'fira/tf/health',
        'fira/lane_confidence',
        'fira/obstacle_severity',
    ]


def test_fira_health_telemetry_inputs_include_drive_state_when_enabled():
    func = _load_function('fira_health_telemetry_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=True,
        FIRA_ENGINE_TF=True,
        FIRA_SAFETY_ARBITER=True,
        FIRA_STATE_MACHINE=True,
    )

    inputs, types = func(cfg)

    assert inputs[-2:] == ['fira/drive/state', 'fira/drive/state_throttle_cap']
    assert types[-2:] == ['str', 'float']


def test_fira_health_telemetry_inputs_include_competition_compliance_channels():
    func = _load_function('fira_health_telemetry_inputs')
    cfg = _Cfg(
        FIRA_ENGINE_YOLO=False,
        FIRA_ENGINE_TF=False,
        FIRA_SAFETY_ARBITER=False,
        FIRA_STATE_MACHINE=False,
        FIRA_COMPETITION_METRICS=True,
    )

    inputs, types = func(cfg)

    assert inputs == [
        'fira/competition/right_lane_score',
        'fira/competition/checkpoint_count',
        'fira/competition/checkpoint_progress',
        'fira/competition/lane_violations',
        'fira/competition/active_frames',
        'fira/competition/compliance_score',
        'fira/competition/compliance_ready',
    ]
    assert types == ['float', 'int', 'float', 'int', 'int', 'float', 'boolean']


def test_fira_challenge_inputs_include_signal_context_when_available():
    func = _load_function('fira_challenge_inputs')
    cfg = _Cfg(FIRA_SAFETY_SIGNAL_ESTIMATOR=True)

    assert func(cfg) == [
        'pilot/angle',
        'pilot/throttle',
        'cam/image_array',
        'fira/lane_confidence',
        'fira/obstacle_severity',
    ]


def test_fira_challenge_inputs_minimal_without_signal_estimator():
    func = _load_function('fira_challenge_inputs')
    cfg = _Cfg(FIRA_SAFETY_SIGNAL_ESTIMATOR=False)

    assert func(cfg) == ['pilot/angle', 'pilot/throttle', 'cam/image_array']


def test_competition_metrics_inputs_include_available_context():
    func = _load_function('fira_competition_metrics_inputs')
    cfg = _Cfg(FIRA_SAFETY_SIGNAL_ESTIMATOR=True, FIRA_STATE_MACHINE=True)

    assert func(cfg) == [
        'fira/lane_confidence',
        'fira/lane_angle',
        'fira/obstacle_severity',
        'fira/drive/state',
    ]


def test_competition_metrics_inputs_minimal_without_sources():
    func = _load_function('fira_competition_metrics_inputs')
    cfg = _Cfg(FIRA_SAFETY_SIGNAL_ESTIMATOR=False, FIRA_STATE_MACHINE=False)

    assert func(cfg) == []


def test_competition_metrics_inputs_include_checkpoint_event_when_detector_enabled():
    func = _load_function('fira_competition_metrics_inputs')
    cfg = _Cfg(
        FIRA_SAFETY_SIGNAL_ESTIMATOR=False,
        FIRA_STATE_MACHINE=False,
        FIRA_CHECKPOINT_DETECTOR_ENABLED=True,
    )

    assert func(cfg) == ['fira/checkpoint_event']


def test_web_controller_inputs_include_fira_channels_when_enabled():
    func = _load_function('web_controller_inputs')
    cfg = _Cfg(
        FIRA_SAFETY_SIGNAL_ESTIMATOR=True,
        FIRA_SAFETY_ARBITER=True,
        FIRA_HEALTH_TELEMETRY=True,
        FIRA_STATE_MACHINE=True,
    )

    assert func(cfg) == [
        'ui/image_array',
        'tub/num_records',
        'user/mode',
        'recording',
        'fira/obstacle_severity',
        'fira/lane_confidence',
        'fira/safety/failsafe_active',
        'fira/safety/lane_weight',
        'fira/safety/obstacle_weight',
        'fira/safety/curve_factor',
        'fira/safety/speed_limit_factor',
        'fira/drive/state',
        'fira/drive/state_throttle_cap',
    ]


def test_web_controller_inputs_are_minimal_without_fira_features():
    func = _load_function('web_controller_inputs')
    cfg = _Cfg(
        FIRA_SAFETY_SIGNAL_ESTIMATOR=False,
        FIRA_SAFETY_ARBITER=False,
        FIRA_HEALTH_TELEMETRY=False,
        FIRA_STATE_MACHINE=False,
    )

    assert func(cfg) == ['ui/image_array', 'tub/num_records', 'user/mode', 'recording']


def test_web_controller_inputs_include_competition_metrics_when_enabled():
    func = _load_function('web_controller_inputs')
    cfg = _Cfg(
        FIRA_SAFETY_SIGNAL_ESTIMATOR=False,
        FIRA_SAFETY_ARBITER=False,
        FIRA_HEALTH_TELEMETRY=False,
        FIRA_STATE_MACHINE=False,
        FIRA_COMPETITION_METRICS=True,
    )

    assert func(cfg) == [
        'ui/image_array',
        'tub/num_records',
        'user/mode',
        'recording',
        'fira/competition/right_lane_score',
        'fira/competition/checkpoint_count',
        'fira/competition/checkpoint_progress',
        'fira/competition/lane_violations',
        'fira/competition/active_frames',
        'fira/competition/compliance_score',
        'fira/competition/compliance_ready',
    ]


def test_health_and_web_inputs_include_checkpoint_channels_when_enabled():
    health_func = _load_function('fira_health_telemetry_inputs')
    web_func = _load_function('web_controller_inputs')

    cfg = _Cfg(
        FIRA_ENGINE_YOLO=False,
        FIRA_ENGINE_TF=False,
        FIRA_SAFETY_ARBITER=False,
        FIRA_STATE_MACHINE=False,
        FIRA_COMPETITION_METRICS=False,
        FIRA_CHECKPOINT_DETECTOR_ENABLED=True,
        FIRA_SAFETY_SIGNAL_ESTIMATOR=False,
        FIRA_HEALTH_TELEMETRY=False,
    )

    inputs, types = health_func(cfg)
    assert inputs == ['fira/checkpoint_event', 'fira/checkpoint_confidence']
    assert types == ['float', 'float']

    web_inputs = web_func(cfg)
    assert web_inputs == [
        'ui/image_array',
        'tub/num_records',
        'user/mode',
        'recording',
        'fira/checkpoint_event',
        'fira/checkpoint_confidence',
    ]
