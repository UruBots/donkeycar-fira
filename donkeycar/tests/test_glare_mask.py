import numpy as np

from donkeycar.parts.cv import ImgGlareMask
from donkeycar.parts.image_transformations import ImageTransformations


class DummyConfig:
    GLARE_MASK = True
    GLARE_MASK_SAT_LOW = 80
    GLARE_MASK_VAL_HIGH = 240
    GLARE_MASK_FILL_WITH = 'mean'
    GLARE_MASK_MORPH_KERNEL = 3
    GLARE_MASK_MORPH_ITERATIONS = 1
    TRANSFORMATIONS = ['GLARE_MASK']


def test_img_glare_mask_replaces_bright_low_saturation_region():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[6:10, 6:10] = np.array([255, 255, 255], dtype=np.uint8)

    masked = ImgGlareMask().run(image)

    assert masked is not None
    assert masked[7, 7].tolist() == [0, 0, 0]
    assert masked[0, 0].tolist() == [0, 0, 0]


def test_image_transformations_can_apply_glare_mask_from_config():
    cfg = DummyConfig()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[6:10, 6:10] = np.array([255, 255, 255], dtype=np.uint8)

    transformed = ImageTransformations(cfg, 'TRANSFORMATIONS').run(image)

    assert transformed is not None
    assert transformed[7, 7].tolist() == [0, 0, 0]
    assert transformed[0, 0].tolist() == [0, 0, 0]
