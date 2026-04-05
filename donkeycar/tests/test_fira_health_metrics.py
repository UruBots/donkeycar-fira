import ast
from pathlib import Path


def _load_fira_health_to_metrics_func():
    complete_path = Path(__file__).resolve().parents[1] / 'templates' / 'complete.py'
    source = complete_path.read_text(encoding='utf-8')
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'fira_health_to_metrics':
            func_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(func_module)
            namespace = {}
            exec(compile(func_module, str(complete_path), 'exec'), namespace)
            return namespace['fira_health_to_metrics']

    raise AssertionError('fira_health_to_metrics not found in complete.py')


def test_fira_health_to_metrics_defaults_when_invalid_input():
    func = _load_fira_health_to_metrics_func()

    assert func(None) == (False, 0, 0, 'unknown', 'unavailable', '')


def test_fira_health_to_metrics_maps_extended_fields():
    func = _load_fira_health_to_metrics_func()

    health = {
        'detector_ready': True,
        'consecutive_detection_errors': 2,
        'max_consecutive_detection_errors': 5,
        'state': 'idle',
        'detector_backend': 'tflite',
        'last_detection_error': 'predict failed',
    }

    assert func(health) == (True, 2, 5, 'idle', 'tflite', 'predict failed')
