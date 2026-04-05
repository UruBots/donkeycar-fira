import time

import numpy as np

from donkeycar.parts.fira_signals_engine import fira_engine_yolo as yolo_mod


def _dummy_img():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_yolo_engine_passthrough_when_detector_unavailable(monkeypatch):
    class _InitFailDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('backend unavailable')

    monkeypatch.setattr(yolo_mod, 'YoloDetect', _InitFailDetector)

    engine = yolo_mod.FIRAEngineYolo(
        model_folder='.',
        yolo_model_name='missing.pt',
        yolo_classes=['Stop'],
        apriltag_hz=15,
        zebra_hz=2,
        top_crop_ratio=0.65,
        yolo_disable_after_errors=2,
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
    assert health['engine'] == 'yolo'
    assert health['detector_ready'] is False
    assert health['detector_backend'] == 'unavailable'
    assert health['consecutive_detection_errors'] == 0
    assert 'backend unavailable' in health['last_detection_error']


def test_yolo_engine_disables_after_consecutive_inference_errors(monkeypatch):
    class _FailingRunDetector:
        def __init__(self, *args, **kwargs):
            self.classes = {'0': 'Stop'}

        def run(self, img):
            raise RuntimeError('predict failed')

    monkeypatch.setattr(yolo_mod, 'YoloDetect', _FailingRunDetector)

    engine = yolo_mod.FIRAEngineYolo(
        model_folder='.',
        yolo_model_name='any.pt',
        yolo_classes=['Stop'],
        apriltag_hz=15,
        zebra_hz=2,
        top_crop_ratio=0.65,
        yolo_disable_after_errors=2,
        debug_visuals=False,
        debug=False,
    )

    assert engine.detector_ready is True

    engine.detect_yolo_signals(_dummy_img(), time.time(), 0.2, 0.1)
    assert engine.detector_ready is True
    assert engine.consecutive_detection_errors == 1

    engine.detect_yolo_signals(_dummy_img(), time.time(), 0.2, 0.1)
    assert engine.detector_ready is False
    assert engine.consecutive_detection_errors == 2

    health = engine.get_health()
    assert health['state'] == 'idle'
    assert health['max_consecutive_detection_errors'] == 2
    assert health['detector_backend'] == 'ultralytics'
    assert health['last_detection_error'] == 'predict failed'
