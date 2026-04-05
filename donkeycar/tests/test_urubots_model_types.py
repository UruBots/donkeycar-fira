import pytest

from donkeycar.config import Config
from donkeycar.parts.keras import (
    KerasPilotNet,
    KerasMLP,
    KerasConfidence,
    KerasViT,
    KerasWorldModel,
    KerasDiffusionPolicy,
    KerasCNNLSTM,
)
from donkeycar.parts.interpreter import TfLite, TensorRT
from donkeycar import utils as dk_utils
from donkeycar.utils import get_model_by_type


@pytest.fixture
def cfg() -> Config:
    cfg = Config()
    cfg.DEFAULT_MODEL_TYPE = 'linear'
    cfg.IMAGE_H = 120
    cfg.IMAGE_W = 160
    cfg.IMAGE_DEPTH = 3
    cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8
    cfg.BEHAVIOR_LIST = ['Left', 'Right']
    cfg.NUM_LOCATIONS = 3
    cfg.SEQUENCE_LENGTH = 3
    cfg.MEM_DEPTH = 0
    cfg.CONFIDENCE_THRESHOLD = 0.5
    cfg.DIFFUSION_STEPS = 50
    cfg.DIFFUSION_INFERENCE_STEPS = 10
    return cfg


@pytest.mark.parametrize(
    'model_type, expected_cls',
    [
        ('pilotnet', KerasPilotNet),
        ('mlp', KerasMLP),
        ('confidence', KerasConfidence),
        ('vit', KerasViT),
        ('world_model', KerasWorldModel),
        ('diffusion_policy', KerasDiffusionPolicy),
        ('cnn_lstm', KerasCNNLSTM),
    ],
)
def test_get_model_by_type_urubots_types(cfg, model_type, expected_cls):
    model = get_model_by_type(model_type, cfg)
    assert isinstance(model, expected_cls)


def test_sequence_sizes_for_temporal_urubots_models(cfg):
    assert get_model_by_type('cnn_lstm', cfg).seq_size() == cfg.SEQUENCE_LENGTH
    assert get_model_by_type('world_model', cfg).seq_size() == cfg.SEQUENCE_LENGTH + 1


@pytest.mark.parametrize(
    'model_type, expected_cls, expected_interpreter_cls',
    [
        ('tflite_pilotnet', KerasPilotNet, TfLite),
        ('tensorrt_mlp', KerasMLP, TensorRT),
        ('tflite_confidence', KerasConfidence, TfLite),
        ('tensorrt_vit', KerasViT, TensorRT),
        ('tflite_world_model', KerasWorldModel, TfLite),
        ('tensorrt_diffusion_policy', KerasDiffusionPolicy, TensorRT),
        ('tflite_cnn_lstm', KerasCNNLSTM, TfLite),
    ],
)
def test_get_model_by_type_urubots_prefixed_backends(
        cfg, model_type, expected_cls, expected_interpreter_cls):
    model = get_model_by_type(model_type, cfg)
    assert isinstance(model, expected_cls)
    assert isinstance(model.interpreter, expected_interpreter_cls)


def test_get_model_by_type_urubots_uses_default_when_none(cfg):
    cfg.DEFAULT_MODEL_TYPE = 'pilotnet'
    model = get_model_by_type(None, cfg)
    assert isinstance(model, KerasPilotNet)


def test_load_model_with_fallback_success_without_fallback(cfg, monkeypatch):
    class _Model:
        def __init__(self):
            self.loaded = False

        def load(self, model_path=None):
            self.loaded = True

    model = _Model()

    def _fake_get_model_by_type(model_type, cfg_obj):
        return model

    monkeypatch.setattr(dk_utils, 'get_model_by_type', _fake_get_model_by_type)

    loaded_model, used_type = dk_utils.load_model_with_fallback(
        'pilotnet', cfg, 'dummy_model_path'
    )

    assert loaded_model is model
    assert used_type == 'pilotnet'
    assert model.loaded is True


def test_load_model_with_fallback_uses_fallback_type(cfg, monkeypatch):
    cfg.MODEL_LOAD_ENABLE_FALLBACK = True
    cfg.MODEL_LOAD_FALLBACK_TYPE = 'linear'

    class _FailingModel:
        def load(self, model_path=None):
            raise RuntimeError('preferred backend failed')

    class _FallbackModel:
        def __init__(self):
            self.loaded = False

        def load(self, model_path=None):
            self.loaded = True

    fallback_model = _FallbackModel()
    calls = []

    def _fake_get_model_by_type(model_type, cfg_obj):
        calls.append(model_type)
        if model_type == 'tensorrt_pilotnet':
            return _FailingModel()
        if model_type == 'linear':
            return fallback_model
        raise AssertionError(f'unexpected model_type {model_type}')

    monkeypatch.setattr(dk_utils, 'get_model_by_type', _fake_get_model_by_type)

    loaded_model, used_type = dk_utils.load_model_with_fallback(
        'tensorrt_pilotnet', cfg, 'dummy_model_path'
    )

    assert calls == ['tensorrt_pilotnet', 'linear']
    assert loaded_model is fallback_model
    assert used_type == 'linear'
    assert fallback_model.loaded is True


def test_load_model_with_fallback_raises_when_disabled(cfg, monkeypatch):
    cfg.MODEL_LOAD_ENABLE_FALLBACK = False
    cfg.MODEL_LOAD_FALLBACK_TYPE = 'linear'

    class _FailingModel:
        def load(self, model_path=None):
            raise RuntimeError('preferred backend failed')

    def _fake_get_model_by_type(model_type, cfg_obj):
        return _FailingModel()

    monkeypatch.setattr(dk_utils, 'get_model_by_type', _fake_get_model_by_type)

    with pytest.raises(RuntimeError, match='preferred backend failed'):
        dk_utils.load_model_with_fallback(
            'tensorrt_pilotnet', cfg, 'dummy_model_path'
        )
