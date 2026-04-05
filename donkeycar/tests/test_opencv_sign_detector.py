import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from donkeycar.parts.fira_modular import FiraModular, detect_stop_line
from donkeycar.parts.opencv_sign_detector import OpenCVSignDetector


def _blank_rgb(height=120, width=160):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_stop_sign():
    img = _blank_rgb()
    points = np.array([
        [60, 28], [84, 28], [112, 56], [112, 80],
        [84, 108], [60, 108], [32, 80], [32, 56],
    ], dtype=np.int32)
    cv2.fillPoly(img, [points], (220, 0, 0))
    return img


def _make_no_entry_sign():
    img = _blank_rgb()
    cv2.circle(img, (80, 60), 30, (220, 0, 0), -1)
    cv2.rectangle(img, (50, 54), (110, 66), (0, 0, 0), -1)
    return img


def _make_blue_circle_arrow(direction='right'):
    img = _blank_rgb()
    cv2.circle(img, (80, 60), 34, (0, 140, 255), -1)
    if direction == 'right':
        cv2.arrowedLine(img, (42, 60), (120, 60), (0, 0, 0), 10, tipLength=0.35)
    elif direction == 'left':
        cv2.arrowedLine(img, (118, 60), (40, 60), (0, 0, 0), 10, tipLength=0.35)
    else:
        cv2.arrowedLine(img, (80, 96), (80, 28), (0, 0, 0), 10, tipLength=0.35)
    return img


def _make_dead_end_sign():
    img = _blank_rgb()
    cv2.rectangle(img, (40, 28), (120, 92), (0, 140, 255), -1)
    cv2.line(img, (80, 36), (80, 86), (0, 0, 0), 10)
    cv2.line(img, (56, 36), (104, 36), (0, 0, 0), 10)
    return img


def _make_tunnel_sign():
    img = _blank_rgb()
    cv2.rectangle(img, (36, 26), (124, 94), (0, 140, 255), -1)
    cv2.rectangle(img, (54, 38), (106, 82), (0, 0, 0), -1)
    cv2.rectangle(img, (48, 32), (112, 88), (0, 0, 0), 2)
    return img


def _make_bridge_sign():
    img = _blank_rgb()
    cv2.rectangle(img, (36, 24), (124, 96), (0, 140, 255), -1)
    cv2.rectangle(img, (48, 60), (64, 92), (0, 0, 0), -1)
    cv2.rectangle(img, (96, 60), (112, 92), (0, 0, 0), -1)
    cv2.ellipse(img, (80, 52), (22, 14), 0, 205, 335, (0, 0, 0), 6)
    return img


def _make_stop_line_frame(y=108, thickness=5):
    img = _blank_rgb()
    cv2.line(img, (10, int(y)), (150, int(y)), (255, 255, 255), int(thickness))
    return img


def _make_plain_blue_sign():
    img = _blank_rgb()
    cv2.circle(img, (80, 60), 34, (0, 140, 255), -1)
    return img


def _add_gaussian_noise(img, rng, sigma=7.0):
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_geometric_jitter(img, rng, max_shift=4, scale_jitter=0.04):
    h, w = img.shape[:2]
    dx = float(rng.integers(-max_shift, max_shift + 1))
    dy = float(rng.integers(-max_shift, max_shift + 1))
    scale = 1.0 + float(rng.uniform(-scale_jitter, scale_jitter))

    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, 0.0, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _apply_partial_occlusion(img, rng, max_cover_ratio=0.22):
    out = img.copy()
    h, w = out.shape[:2]
    cover = float(rng.uniform(0.10, max_cover_ratio))
    side = rng.integers(0, 4)

    if side == 0:
        cw = max(1, int(w * cover))
        out[:, :cw] = 0
    elif side == 1:
        cw = max(1, int(w * cover))
        out[:, w - cw:] = 0
    elif side == 2:
        ch = max(1, int(h * cover))
        out[:ch, :] = 0
    else:
        ch = max(1, int(h * cover))
        out[h - ch:, :] = 0

    return out


def _adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) * float(factor)
    hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_motion_blur(img, rng, kernel_size=7):
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = np.zeros((k, k), dtype=np.float32)
    if bool(rng.integers(0, 2)):
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k

    return cv2.filter2D(img, -1, kernel)


def _rescale_frame(img, scale):
    h, w = img.shape[:2]
    nw = max(32, int(round(w * float(scale))))
    nh = max(24, int(round(h * float(scale))))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_color_cast(img, rgb_scales=(1.0, 1.0, 1.0)):
    scales = np.asarray(rgb_scales, dtype=np.float32).reshape((1, 1, 3))
    out = img.astype(np.float32) * scales
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_jpeg_artifacts(img, quality=45):
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img.copy()
    bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_detect_stop_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_stop_sign())

    assert result is not None
    assert result.tag_id == -1
    assert result.label == 'STOP'
    assert result.confidence >= 0.45


def test_detect_no_entry_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_no_entry_sign())

    assert result is not None
    assert result.tag_id == -1
    assert result.label == 'NO_ENTRY'
    assert result.confidence >= 0.45


def test_detect_turn_right_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_blue_circle_arrow('right'))

    assert result is not None
    assert result.tag_id == 1
    assert result.label == 'TURN_RIGHT'


def test_detect_turn_left_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_blue_circle_arrow('left'))

    assert result is not None
    assert result.tag_id == 3
    assert result.label == 'TURN_LEFT'


def test_detect_forward_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_blue_circle_arrow('forward'))

    assert result is not None
    assert result.tag_id == 0
    assert result.label == 'FORWARD'


def test_detect_dead_end_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_dead_end_sign())

    assert result is not None
    assert result.tag_id == 4
    assert result.label == 'DEAD_END'


def test_detect_tunnel_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_tunnel_sign())

    assert result is not None
    assert result.tag_id == 6
    assert result.label == 'TUNNEL'


def test_detect_bridge_sign():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_bridge_sign())

    assert result is not None
    assert result.tag_id == 7
    assert result.label == 'BRIDGE'


def test_detect_plain_blue_sign_returns_none():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    result = detector.detect(_make_plain_blue_sign())

    assert result is None


def test_fira_modular_falls_back_to_opencv_signs(monkeypatch):
    class NoTags:
        def detect(self, gray_img):
            return []

    modular = FiraModular(
        tag_dict={-1: 'STOP', 0: 'FORWARD', 1: 'TURN_RIGHT', 3: 'TURN_LEFT', 4: 'DEAD_END', 6: 'TUNNEL', 7: 'BRIDGE'},
        camera_to_front=0.10,
        vehicle_width=0.12,
        vehicle_length=0.25,
        debug=False,
        require_zebra=False,
        tag_detection_hz=5,
        ejecutar_accion_al_dejar_de_ver_tag=False,
        usar_una_sola_camara=True,
        zebra_detection_hz=5,
        tag_ratio_threshold=26,
        reescalar_tag_img=False,
        usar_thread_tag=False,
        opencv_sign_fallback=True,
        opencv_sign_min_confidence=0.45,
        opencv_sign_min_area=150,
        opencv_sign_debug=False,
    )
    modular.tag_detector = NoTags()

    tag = modular.detect_apriltag(_make_stop_sign())

    assert tag is not None
    assert tag.tag_id == -1


def test_fira_modular_falls_back_to_tunnel_signs():
    class NoTags:
        def detect(self, gray_img):
            return []

    modular = FiraModular(
        tag_dict={-1: 'STOP', 0: 'FORWARD', 1: 'TURN_RIGHT', 3: 'TURN_LEFT', 4: 'DEAD_END', 6: 'TUNNEL', 7: 'BRIDGE'},
        camera_to_front=0.10,
        vehicle_width=0.12,
        vehicle_length=0.25,
        debug=False,
        require_zebra=False,
        tag_detection_hz=5,
        ejecutar_accion_al_dejar_de_ver_tag=False,
        usar_una_sola_camara=True,
        zebra_detection_hz=5,
        tag_ratio_threshold=26,
        reescalar_tag_img=False,
        usar_thread_tag=False,
        opencv_sign_fallback=True,
        opencv_sign_min_confidence=0.45,
        opencv_sign_min_area=150,
        opencv_sign_debug=False,
    )
    modular.tag_detector = NoTags()

    tag = modular.detect_apriltag(_make_tunnel_sign())

    assert tag is not None
    assert tag.tag_id == 6


def test_fira_modular_does_not_use_opencv_when_fallback_disabled():
    class NoTags:
        def detect(self, gray_img):
            return []

    modular = FiraModular(
        tag_dict={-1: 'STOP', 0: 'FORWARD', 1: 'TURN_RIGHT', 3: 'TURN_LEFT', 4: 'DEAD_END', 6: 'TUNNEL', 7: 'BRIDGE'},
        camera_to_front=0.10,
        vehicle_width=0.12,
        vehicle_length=0.25,
        debug=False,
        require_zebra=False,
        tag_detection_hz=5,
        ejecutar_accion_al_dejar_de_ver_tag=False,
        usar_una_sola_camara=True,
        zebra_detection_hz=5,
        tag_ratio_threshold=26,
        reescalar_tag_img=False,
        usar_thread_tag=False,
        opencv_sign_fallback=False,
        opencv_sign_min_confidence=0.45,
        opencv_sign_min_area=150,
        opencv_sign_debug=False,
    )
    modular.tag_detector = NoTags()

    tag = modular.detect_apriltag(_make_stop_sign())

    assert tag is None


def test_fira_modular_falls_back_to_bridge_signs():
    class NoTags:
        def detect(self, gray_img):
            return []

    modular = FiraModular(
        tag_dict={-1: 'STOP', 0: 'FORWARD', 1: 'TURN_RIGHT', 3: 'TURN_LEFT', 4: 'DEAD_END', 6: 'TUNNEL', 7: 'BRIDGE'},
        camera_to_front=0.10,
        vehicle_width=0.12,
        vehicle_length=0.25,
        debug=False,
        require_zebra=False,
        tag_detection_hz=5,
        ejecutar_accion_al_dejar_de_ver_tag=False,
        usar_una_sola_camara=True,
        zebra_detection_hz=5,
        tag_ratio_threshold=26,
        reescalar_tag_img=False,
        usar_thread_tag=False,
        opencv_sign_fallback=True,
        opencv_sign_min_confidence=0.45,
        opencv_sign_min_area=150,
        opencv_sign_debug=False,
    )
    modular.tag_detector = NoTags()

    tag = modular.detect_apriltag(_make_bridge_sign())

    assert tag is not None
    assert tag.tag_id == 7


def test_detect_stop_line_near_bottom_returns_distance():
    has_stop_line, dist = detect_stop_line(_make_stop_line_frame(y=108, thickness=5))

    assert has_stop_line
    assert dist is not None
    assert 7 <= dist <= 15


def test_detect_stop_line_absent_on_blank_frame():
    has_stop_line, dist = detect_stop_line(_blank_rgb())

    assert not has_stop_line
    assert dist is None


def test_tunnel_detection_is_stable_across_noisy_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_tunnel_sign()
    rng = np.random.default_rng(1234)

    labels = []
    for _ in range(20):
        frame = _add_gaussian_noise(base, rng, sigma=7.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('TUNNEL') >= 16
    assert labels.count('DEAD_END') <= 2


def test_dead_end_detection_is_stable_across_noisy_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_dead_end_sign()
    rng = np.random.default_rng(5678)

    labels = []
    for _ in range(20):
        frame = _add_gaussian_noise(base, rng, sigma=7.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('DEAD_END') >= 16
    assert labels.count('TUNNEL') <= 2


def test_tunnel_detection_is_stable_across_jittered_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_tunnel_sign()
    rng = np.random.default_rng(2468)

    labels = []
    for _ in range(20):
        frame = _apply_geometric_jitter(base, rng, max_shift=4, scale_jitter=0.04)
        frame = _add_gaussian_noise(frame, rng, sigma=5.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('TUNNEL') >= 15
    assert labels.count('DEAD_END') <= 3


def test_dead_end_detection_is_stable_across_jittered_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_dead_end_sign()
    rng = np.random.default_rng(1357)

    labels = []
    for _ in range(20):
        frame = _apply_geometric_jitter(base, rng, max_shift=4, scale_jitter=0.04)
        frame = _add_gaussian_noise(frame, rng, sigma=5.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('DEAD_END') >= 15
    assert labels.count('TUNNEL') <= 3


def test_tunnel_detection_is_stable_across_partially_occluded_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_tunnel_sign()
    rng = np.random.default_rng(2026)

    labels = []
    for _ in range(20):
        frame = _apply_partial_occlusion(base, rng, max_cover_ratio=0.22)
        frame = _add_gaussian_noise(frame, rng, sigma=5.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('TUNNEL') >= 14
    assert labels.count('DEAD_END') <= 4


def test_dead_end_detection_is_stable_across_partially_occluded_frames():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    base = _make_dead_end_sign()
    rng = np.random.default_rng(2027)

    labels = []
    for _ in range(20):
        frame = _apply_partial_occlusion(base, rng, max_cover_ratio=0.22)
        frame = _add_gaussian_noise(frame, rng, sigma=5.0)
        result = detector.detect(frame)
        labels.append(None if result is None else result.label)

    assert labels.count('DEAD_END') >= 14
    assert labels.count('TUNNEL') <= 4


def test_alternating_tunnel_dead_end_sequence_has_low_confusion():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(3141)

    expected_labels = []
    predicted_labels = []
    for i in range(30):
        if i % 2 == 0:
            base = _make_tunnel_sign()
            expected = 'TUNNEL'
        else:
            base = _make_dead_end_sign()
            expected = 'DEAD_END'

        frame = _apply_geometric_jitter(base, rng, max_shift=3, scale_jitter=0.03)
        frame = _apply_partial_occlusion(frame, rng, max_cover_ratio=0.18)
        frame = _add_gaussian_noise(frame, rng, sigma=4.0)
        result = detector.detect(frame)

        expected_labels.append(expected)
        predicted_labels.append(None if result is None else result.label)

    correct = sum(1 for exp, pred in zip(expected_labels, predicted_labels) if exp == pred)
    wrong_cross = sum(
        1
        for exp, pred in zip(expected_labels, predicted_labels)
        if (exp == 'TUNNEL' and pred == 'DEAD_END') or (exp == 'DEAD_END' and pred == 'TUNNEL')
    )

    assert correct >= 23
    assert wrong_cross <= 5


def test_block_switch_sequence_recovers_after_class_change():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(2718)

    sequence = ['TUNNEL'] * 12 + ['DEAD_END'] * 12
    predicted = []
    for label in sequence:
        base = _make_tunnel_sign() if label == 'TUNNEL' else _make_dead_end_sign()
        frame = _apply_geometric_jitter(base, rng, max_shift=3, scale_jitter=0.03)
        frame = _add_gaussian_noise(frame, rng, sigma=4.0)
        result = detector.detect(frame)
        predicted.append(None if result is None else result.label)

    # Around the switch, detector should settle to the new class quickly.
    post_switch_window = predicted[12:16]
    assert post_switch_window.count('DEAD_END') >= 3


def test_tunnel_detection_survives_extreme_lighting_conditions():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(9090)
    base = _make_tunnel_sign()

    low_light = _adjust_brightness(base, 0.72)
    high_light = _adjust_brightness(base, 1.35)

    low_labels = []
    high_labels = []
    for _ in range(12):
        low_frame = _add_gaussian_noise(low_light, rng, sigma=4.0)
        high_frame = _add_gaussian_noise(high_light, rng, sigma=4.0)
        low_res = detector.detect(low_frame)
        high_res = detector.detect(high_frame)
        low_labels.append(None if low_res is None else low_res.label)
        high_labels.append(None if high_res is None else high_res.label)

    assert low_labels.count('TUNNEL') >= 10
    assert low_labels.count('DEAD_END') <= 1
    assert high_labels.count('TUNNEL') >= 10
    assert high_labels.count('DEAD_END') <= 1


def test_dead_end_detection_survives_extreme_lighting_conditions():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(9191)
    base = _make_dead_end_sign()

    low_light = _adjust_brightness(base, 0.72)
    high_light = _adjust_brightness(base, 1.35)

    low_labels = []
    high_labels = []
    for _ in range(12):
        low_frame = _add_gaussian_noise(low_light, rng, sigma=4.0)
        high_frame = _add_gaussian_noise(high_light, rng, sigma=4.0)
        low_res = detector.detect(low_frame)
        high_res = detector.detect(high_frame)
        low_labels.append(None if low_res is None else low_res.label)
        high_labels.append(None if high_res is None else high_res.label)

    assert low_labels.count('DEAD_END') >= 10
    assert low_labels.count('TUNNEL') <= 1
    assert high_labels.count('DEAD_END') >= 10
    assert high_labels.count('TUNNEL') <= 1


def test_tunnel_detection_is_stable_with_motion_blur_sequence():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(7771)
    base = _make_tunnel_sign()

    labels = []
    for _ in range(20):
        frame = _apply_geometric_jitter(base, rng, max_shift=2, scale_jitter=0.02)
        frame = _apply_motion_blur(frame, rng, kernel_size=7)
        frame = _add_gaussian_noise(frame, rng, sigma=3.5)
        res = detector.detect(frame)
        labels.append(None if res is None else res.label)

    assert labels.count('TUNNEL') >= 15
    assert labels.count('DEAD_END') <= 3


def test_dead_end_detection_is_stable_with_motion_blur_sequence():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(7772)
    base = _make_dead_end_sign()

    labels = []
    for _ in range(20):
        frame = _apply_geometric_jitter(base, rng, max_shift=2, scale_jitter=0.02)
        frame = _apply_motion_blur(frame, rng, kernel_size=7)
        frame = _add_gaussian_noise(frame, rng, sigma=3.5)
        res = detector.detect(frame)
        labels.append(None if res is None else res.label)

    assert labels.count('DEAD_END') >= 15
    assert labels.count('TUNNEL') <= 3


def test_tunnel_detection_is_stable_across_resolution_scales():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(8081)
    base = _make_tunnel_sign()

    labels = []
    for scale in [0.55, 0.7, 0.85, 1.0, 1.2]:
        for _ in range(5):
            frame = _rescale_frame(base, scale)
            frame = _apply_geometric_jitter(frame, rng, max_shift=2, scale_jitter=0.01)
            frame = _add_gaussian_noise(frame, rng, sigma=3.0)
            res = detector.detect(frame)
            labels.append(None if res is None else res.label)

    assert labels.count('TUNNEL') >= 20
    assert labels.count('DEAD_END') <= 3


def test_dead_end_detection_is_stable_across_resolution_scales():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(8082)
    base = _make_dead_end_sign()

    labels = []
    for scale in [0.55, 0.7, 0.85, 1.0, 1.2]:
        for _ in range(5):
            frame = _rescale_frame(base, scale)
            frame = _apply_geometric_jitter(frame, rng, max_shift=2, scale_jitter=0.01)
            frame = _add_gaussian_noise(frame, rng, sigma=3.0)
            res = detector.detect(frame)
            labels.append(None if res is None else res.label)

    assert labels.count('DEAD_END') >= 20
    assert labels.count('TUNNEL') <= 3


def test_tunnel_detection_is_stable_with_color_cast_and_compression():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(8181)
    base = _make_tunnel_sign()

    labels = []
    casts = [(1.10, 0.95, 0.95), (0.95, 1.05, 1.10), (1.00, 1.00, 0.90)]
    for i in range(18):
        frame = _apply_color_cast(base, casts[i % len(casts)])
        frame = _apply_jpeg_artifacts(frame, quality=45)
        frame = _add_gaussian_noise(frame, rng, sigma=3.0)
        res = detector.detect(frame)
        labels.append(None if res is None else res.label)

    assert labels.count('TUNNEL') >= 15
    assert labels.count('DEAD_END') <= 2


def test_dead_end_detection_is_stable_with_color_cast_and_compression():
    detector = OpenCVSignDetector(min_area=150, min_confidence=0.45)
    rng = np.random.default_rng(8182)
    base = _make_dead_end_sign()

    labels = []
    casts = [(1.10, 0.95, 0.95), (0.95, 1.05, 1.10), (1.00, 1.00, 0.90)]
    for i in range(18):
        frame = _apply_color_cast(base, casts[i % len(casts)])
        frame = _apply_jpeg_artifacts(frame, quality=45)
        frame = _add_gaussian_noise(frame, rng, sigma=3.0)
        res = detector.detect(frame)
        labels.append(None if res is None else res.label)

    assert labels.count('DEAD_END') >= 15
    assert labels.count('TUNNEL') <= 2