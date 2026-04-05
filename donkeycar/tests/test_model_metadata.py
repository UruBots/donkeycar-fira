from donkeycar.config import Config
from donkeycar import utils


class DummyPilot:
    def __init__(self):
        self.loaded = None

    def load(self, model_path):
        self.loaded = model_path


def test_model_metadata_round_trip(tmp_path):
    model_path = tmp_path / 'pilot.h5'
    model_path.write_text('stub')
    metadata = {
        'use_glare_mask': True,
        'use_style_transfer': False,
        'style_transfer_preset': 'random',
        'style_transfer_blend': 0.35,
    }

    utils.write_model_metadata(str(model_path), metadata)

    loaded = utils.read_model_metadata(str(model_path))

    assert loaded == metadata


def test_load_model_with_fallback_applies_glare_metadata(monkeypatch, tmp_path):
    model_path = tmp_path / 'pilot.h5'
    model_path.write_text('stub')
    utils.write_model_metadata(str(model_path), {
        'use_glare_mask': True,
        'use_style_transfer': True,
        'style_transfer_preset': 'sepia',
        'style_transfer_blend': 1.0,
    })

    cfg = Config()
    cfg.DEFAULT_MODEL_TYPE = 'linear'
    cfg.GLARE_MASK = False
    cfg.TRANSFORMATIONS = ['CROP']
    cfg.AUG_STYLE_TRANSFER = False
    cfg.AUGMENTATIONS = ['BRIGHTNESS']
    cfg.MODEL_LOAD_ENABLE_FALLBACK = False

    dummy = DummyPilot()

    def fake_get_model_by_type(model_type, config):
        return dummy

    monkeypatch.setattr(utils, 'get_model_by_type', fake_get_model_by_type)

    kl, selected_model_type = utils.load_model_with_fallback('linear', cfg, str(model_path))

    assert kl is dummy
    assert selected_model_type == 'linear'
    assert dummy.loaded == str(model_path)
    assert cfg.GLARE_MASK is True
    assert cfg.TRANSFORMATIONS[0] == 'GLARE_MASK'
    assert 'CROP' in cfg.TRANSFORMATIONS
    assert cfg.AUG_STYLE_TRANSFER is True
    assert cfg.AUG_STYLE_TRANSFER_PRESET == 'sepia'
    assert cfg.AUG_STYLE_TRANSFER_BLEND == 1.0
    assert cfg.AUGMENTATIONS == ['BRIGHTNESS', 'STYLE_TRANSFER']
