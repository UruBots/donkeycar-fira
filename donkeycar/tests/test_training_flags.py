from donkeycar.config import Config
from donkeycar.pipeline.training import apply_training_augmentation_overrides


def test_apply_training_augmentation_overrides_enables_glare_and_style():
    cfg = Config()
    cfg.GLARE_MASK = False
    cfg.TRANSFORMATIONS = ['CROP']
    cfg.AUG_STYLE_TRANSFER = False
    cfg.AUGMENTATIONS = ['BRIGHTNESS']

    apply_training_augmentation_overrides(
        cfg,
        mask_glare=True,
        style_transfer=True,
        style_transfer_preset='sepia',
        style_transfer_blend=0.75,
    )

    assert cfg.GLARE_MASK is True
    assert cfg.TRANSFORMATIONS == ['GLARE_MASK', 'CROP']
    assert cfg.AUG_STYLE_TRANSFER is True
    assert cfg.AUG_STYLE_TRANSFER_PRESET == 'sepia'
    assert cfg.AUG_STYLE_TRANSFER_BLEND == 0.75
    assert cfg.AUGMENTATIONS == ['BRIGHTNESS', 'STYLE_TRANSFER']


def test_apply_training_augmentation_overrides_can_disable_flags():
    cfg = Config()
    cfg.GLARE_MASK = True
    cfg.TRANSFORMATIONS = ['GLARE_MASK', 'CROP']
    cfg.AUG_STYLE_TRANSFER = True
    cfg.AUGMENTATIONS = ['BRIGHTNESS', 'STYLE_TRANSFER']

    apply_training_augmentation_overrides(
        cfg,
        mask_glare=False,
        style_transfer=False,
    )

    assert cfg.GLARE_MASK is False
    assert cfg.TRANSFORMATIONS == ['CROP']
    assert cfg.AUG_STYLE_TRANSFER is False
    assert cfg.AUGMENTATIONS == ['BRIGHTNESS']
