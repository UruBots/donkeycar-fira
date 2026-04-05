import albumentations.core.transforms_interface
import logging
import albumentations as A
import cv2
import numpy as np
from albumentations import GaussianBlur
from albumentations.augmentations.transforms import RandomBrightnessContrast

from donkeycar.config import Config


logger = logging.getLogger(__name__)


class ImgStyleTransfer(A.ImageOnlyTransform):
    """Synthetic style transfer augmentation for training images.

    This transform keeps the scene geometry intact while shifting color
    palettes and texture statistics to emulate different lighting and surface
    styles.
    """

    STYLE_PRESETS = {
        'warm': {
            'matrix': np.array([
                [1.03, 0.01, -0.04],
                [0.00, 1.00, -0.02],
                [-0.02, 0.01, 0.98],
            ], dtype=np.float32),
            'saturation': 1.10,
            'value': 1.04,
            'gamma': 0.96,
        },
        'cool': {
            'matrix': np.array([
                [0.96, 0.01, 0.03],
                [0.00, 1.00, 0.02],
                [0.04, 0.00, 1.04],
            ], dtype=np.float32),
            'saturation': 0.95,
            'value': 1.01,
            'gamma': 1.02,
        },
        'desaturated': {
            'matrix': np.array([
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.01, 0.98],
            ], dtype=np.float32),
            'saturation': 0.72,
            'value': 1.00,
            'gamma': 1.00,
        },
        'high_contrast': {
            'matrix': np.array([
                [1.10, -0.02, -0.04],
                [-0.01, 1.06, -0.02],
                [-0.02, -0.01, 1.12],
            ], dtype=np.float32),
            'saturation': 1.05,
            'value': 1.08,
            'gamma': 0.90,
        },
        'sepia': {
            'matrix': np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ], dtype=np.float32),
            'saturation': 0.82,
            'value': 1.00,
            'gamma': 1.00,
        },
    }

    def __init__(self, preset='random', blend=0.35, p=0.5, always_apply=False):
        super().__init__(always_apply=always_apply, p=p)
        self.preset = preset
        self.blend = blend

    def get_transform_init_args_names(self):
        return ('preset', 'blend')

    def _select_preset(self):
        if self.preset != 'random':
            return self.preset
        return np.random.choice(list(self.STYLE_PRESETS.keys()))

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        gamma = max(gamma, 1e-6)
        inverse_gamma = 1.0 / gamma
        table = np.array([
            ((index / 255.0) ** inverse_gamma) * 255.0
            for index in range(256)
        ], dtype=np.uint8)
        return table

    def _apply_preset(self, image: np.ndarray, preset_name: str) -> np.ndarray:
        preset = self.STYLE_PRESETS[preset_name]
        working = image.astype(np.float32)

        stylized = np.tensordot(working, preset['matrix'].T, axes=1)
        stylized = np.clip(stylized, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(stylized, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * preset['saturation'], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * preset['value'], 0, 255)
        stylized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if preset['gamma'] != 1.0:
            stylized = cv2.LUT(stylized, self._gamma_lut(preset['gamma']))

        if preset_name in ('high_contrast', 'sepia'):
            sharpen_kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ], dtype=np.float32)
            stylized = cv2.filter2D(stylized, -1, sharpen_kernel)

        return stylized

    def apply(self, image, **params):
        preset_name = self._select_preset()
        stylized = self._apply_preset(image, preset_name)
        blend = self.blend
        if isinstance(blend, (tuple, list)) and len(blend) == 2:
            blend = float(np.random.uniform(blend[0], blend[1]))
        blend = float(np.clip(blend, 0.0, 1.0))
        return cv2.addWeighted(image, 1.0 - blend, stylized, blend, 0)

    def run(self, image):
        return self.apply(image)


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5, always_apply=False):
        aug_list = getattr(cfg, key, [])
        augmentations = [ImageAugmentation.create(a, cfg, prob, always_apply)
                         for a in aug_list]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(cls, aug_type: str, config: Config, prob, always) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmentation factory. Cropping and trapezoidal mask are
            transformations which should be applied in training, validation
            and inference. Multiply, Blur and similar are augmentations
            which should be used only in training. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            logger.info(f'Creating augmentation {aug_type} {b_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=b_limit,
                                            p=prob, always_apply=always)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob, always_apply=always)
        elif aug_type == 'STYLE_TRANSFER':
            preset = getattr(config, 'AUG_STYLE_TRANSFER_PRESET', 'random')
            blend = getattr(config, 'AUG_STYLE_TRANSFER_BLEND', 0.35)
            logger.info(f'Creating augmentation {aug_type} preset={preset} blend={blend}')
            return ImgStyleTransfer(preset=preset, blend=blend,
                                    p=prob, always_apply=always)

    # Parts interface
    def run(self, img_arr):
        if len(self.augmentations) == 0:
            return img_arr
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr

