import numpy as np
import pytest

from donkeycar.parts.fira_signals_engine import fira_engine_tensorflow as tf_mod


def _dummy_img():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_tf_engine_passthrough_when_detector_unavailable(monkeypatch):
    class _InitFailDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('backend unavailable')

    monkeypatch.setattr(tf_mod, 'TensorflowDetect', _InitFailDetector)

    engine = tf_mod.FIRAEngineTensorFlow(
        model_folder='.',
        tf_model_name='missing_model',
        fira_classes={1: 'Stop'},
        apriltag_hz=15,
        zebra_hz=2,
        top_crop_ratio=0.65,
        tf_disable_after_errors=2,
        debug_visuals=False,
        debug=False,
    )

    angle, throttle = 0.11, 0.22
    out_angle, out_throttle, out_img = engine.run(angle, throttle, _dummy_img())

    assert engine.detector_ready is False
    assert out_angle == angle
    assert out_throttle == throttle
    assert out_img.shape == _dummy_img().shape

    health = engine.get_health()
    assert health['engine'] == 'tensorflow'
    assert health['detector_ready'] is False
    assert health['detector_backend'] == 'unavailable'
    assert health['consecutive_detection_errors'] == 0
    assert 'backend unavailable' in health['last_detection_error']


def test_tf_engine_disables_after_consecutive_inference_errors(monkeypatch):
    class _FailingRunDetector:
        def __init__(self, *args, **kwargs):
            self.classes = {1: 'Stop'}

        def run(self, img):
            raise RuntimeError('predict failed')

    monkeypatch.setattr(tf_mod, 'TensorflowDetect', _FailingRunDetector)

    engine = tf_mod.FIRAEngineTensorFlow(
        model_folder='.',
        tf_model_name='any_model',
        fira_classes={1: 'Stop'},
        apriltag_hz=15,
        zebra_hz=2,
        top_crop_ratio=0.65,
        tf_disable_after_errors=2,
        debug_visuals=False,
        debug=False,
    )

    assert engine.detector_ready is True

    engine.detect_tf_signals(_dummy_img(), 0.0, 0.2, 0.1)
    assert engine.detector_ready is True
    assert engine.consecutive_detection_errors == 1

    engine.detect_tf_signals(_dummy_img(), 0.0, 0.2, 0.1)
    assert engine.detector_ready is False
    assert engine.consecutive_detection_errors == 2

    health = engine.get_health()
    assert health['state'] == 'idle'
    assert health['max_consecutive_detection_errors'] == 2
    assert health['detector_backend'] == 'unavailable'
    assert health['last_detection_error'] == 'predict failed'


def test_tensorflow_detect_uses_saved_model_backend(monkeypatch):
    class _FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value)

        def __getitem__(self, idx):
            return _FakeTensor(self._value[idx])

        def numpy(self):
            return self._value

    class _SavedModel:
        def __init__(self):
            self.signatures = {
                'serving_default': self._detect,
            }

        def _detect(self, _input_tensor):
            return {
                'detection_boxes': _FakeTensor([[[0.1, 0.2, 0.3, 0.4]]]),
                'detection_classes': _FakeTensor([[1.0]]),
                'detection_scores': _FakeTensor([[0.9]]),
            }

    monkeypatch.setattr(tf_mod.tf, 'convert_to_tensor', lambda value: _FakeTensor(value))
    monkeypatch.setattr(tf_mod.tf.saved_model, 'load', lambda _path: _SavedModel())

    detector = tf_mod.TensorflowDetect('.', 'saved_model_dir')
    boxes, class_ids, scores = detector.run(_dummy_img())

    assert detector.backend == 'saved_model'
    assert boxes.shape == (1, 4)
    assert class_ids.tolist() == [1]
    assert scores.tolist() == pytest.approx([0.9])


def test_tensorflow_detect_uses_tflite_backend(monkeypatch):
    class _FakeInterpreter:
        def __init__(self, model_path=None):
            self.model_path = model_path
            self._input = None

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [{'index': 0, 'shape': np.array([1, 120, 160, 3]), 'dtype': np.uint8}]

        def get_output_details(self):
            return [{'index': 1}, {'index': 2}, {'index': 3}, {'index': 4}]

        def set_tensor(self, index, value):
            assert index == 0
            self._input = value

        def invoke(self):
            return None

        def get_tensor(self, index):
            outputs = {
                1: np.array([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.5, 0.6]]], dtype=np.float32),
                2: np.array([[1, 4]], dtype=np.float32),
                3: np.array([[0.95, 0.7]], dtype=np.float32),
                4: np.array([2], dtype=np.float32),
            }
            return outputs[index]

    monkeypatch.setattr(tf_mod.tf.lite, 'Interpreter', _FakeInterpreter)

    detector = tf_mod.TensorflowDetect('.', 'model.tflite')
    boxes, class_ids, scores = detector.run(_dummy_img())

    assert detector.backend == 'tflite'
    assert boxes.shape == (2, 4)
    assert class_ids.tolist() == [1, 4]
    assert scores.tolist() == pytest.approx([0.95, 0.7])
