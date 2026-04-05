import numpy as np

from donkeycar.pipeline.augmentations import ImageAugmentation, ImgStyleTransfer


class DummyConfig:
    AUG_STYLE_TRANSFER = True
    AUG_STYLE_TRANSFER_PRESET = 'sepia'
    AUG_STYLE_TRANSFER_BLEND = 1.0
    AUGMENTATIONS = ['STYLE_TRANSFER']


def test_img_style_transfer_changes_rgb_image():
    image = np.full((16, 16, 3), [30, 80, 200], dtype=np.uint8)

    transformed = ImgStyleTransfer(preset='sepia', blend=1.0, p=1.0, always_apply=True).run(image)

    assert transformed is not None
    assert transformed.shape == image.shape
    assert transformed.dtype == np.uint8
    assert not np.array_equal(transformed, image)


def test_image_augmentation_can_apply_style_transfer_from_config():
    cfg = DummyConfig()
    image = np.full((16, 16, 3), [30, 80, 200], dtype=np.uint8)

    transformed = ImageAugmentation(cfg, 'AUGMENTATIONS', prob=1.0, always_apply=True).run(image)

    assert transformed is not None
    assert transformed.shape == image.shape
    assert transformed.dtype == np.uint8
    assert not np.array_equal(transformed, image)
