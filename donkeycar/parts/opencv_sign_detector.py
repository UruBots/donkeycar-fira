from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectedSign:
    tag_id: int
    label: str
    confidence: float
    corners: np.ndarray


class OpenCVSignDetector:
    """Pure OpenCV fallback detector for FIRA traffic-style signs."""

    ACTION_TO_TAG_ID = {
        'STOP': -1,
        'DEAD_END': 4,
        'FORWARD': 0,
        'TURN_RIGHT': 1,
        'TURN_LEFT': 3,
        'TUNNEL': 6,
        'BRIDGE': 7,
    }

    def __init__(
        self,
        min_area: int = 220,
        min_confidence: float = 0.55,
        image_is_rgb: bool = True,
        debug: bool = False,
    ) -> None:
        self.min_area = int(min_area)
        self.min_confidence = float(min_confidence)
        self.image_is_rgb = bool(image_is_rgb)
        self.debug = bool(debug)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _to_hsv(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV if self.image_is_rgb else cv2.COLOR_BGR2HSV)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if self.image_is_rgb else cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _corners_from_contour(contour: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
        return np.intp(corners)

    def _build_detection(self, tag_id: int, label: str, confidence: float, contour: np.ndarray) -> DetectedSign:
        corners = self._corners_from_contour(contour)
        return DetectedSign(tag_id=tag_id, label=label, confidence=float(confidence), corners=corners)

    def _score_contour(self, contour: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(max(1, w * h))
        mask_area = float(np.count_nonzero(mask[y:y + h, x:x + w]))
        fill_ratio = self._clip(area / box_area, 0.0, 1.0)
        color_ratio = self._clip(mask_area / box_area, 0.0, 1.0)
        return area, fill_ratio, color_ratio

    def _detect_red_sign(self, img: np.ndarray, hsv: np.ndarray) -> Optional[DetectedSign]:
        lower1 = np.array([0, 70, 70], dtype=np.uint8)
        upper1 = np.array([12, 255, 255], dtype=np.uint8)
        lower2 = np.array([168, 70, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        if len(contours) >= 2:
            ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            boxes = [cv2.boundingRect(contour) for contour in ranked]
            x0 = min(box[0] for box in boxes)
            y0 = min(box[1] for box in boxes)
            x1 = max(box[0] + box[2] for box in boxes)
            y1 = max(box[1] + box[3] for box in boxes)
            width = x1 - x0
            height = y1 - y0
            if width > 0 and height > 0:
                x_spread = max(box[0] for box in boxes) - min(box[0] for box in boxes)
                y_spread = max(box[1] for box in boxes) - min(box[1] for box in boxes)
                if x_spread < width * 0.18 and y_spread > height * 0.35:
                    confidence = 0.78
                    contour = np.vstack(ranked)
                    if confidence >= self.min_confidence:
                        return self._build_detection(self.ACTION_TO_TAG_ID['STOP'], 'NO_ENTRY', confidence, contour)

        contour = max(contours, key=cv2.contourArea)

        area, fill_ratio, color_ratio = self._score_contour(contour, mask)
        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y + h, x:x + w]
        roi_hsv = self._to_hsv(roi)
        black_mask = cv2.inRange(roi_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 90, 90], dtype=np.uint8))
        mid_y = max(1, h // 2)
        band_height = max(1, h // 5)
        x_margin = max(1, w // 5)
        band = black_mask[mid_y - band_height:mid_y + band_height, x_margin:w - x_margin]
        top_band = black_mask[:band_height, x_margin:w - x_margin]
        bottom_band = black_mask[-band_height:, x_margin:w - x_margin]
        band_ratio = float(np.mean(band)) if band.size else 0.0
        edge_band_ratio = max(
            float(np.mean(top_band)) if top_band.size else 0.0,
            float(np.mean(bottom_band)) if bottom_band.size else 0.0,
        )
        band_contrast = band_ratio / max(1.0, edge_band_ratio)

        perimeter = max(1.0, cv2.arcLength(contour, True))
        circularity = self._clip((4.0 * np.pi * area) / (perimeter * perimeter), 0.0, 1.0)

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        aspect = max(w, h) / max(1.0, min(w, h))

        if vertices >= 6 and aspect < 1.45:
            confidence = self._clip(0.30 + 0.40 * fill_ratio + 0.30 * circularity, 0.0, 1.0)
            if confidence >= self.min_confidence:
                return self._build_detection(self.ACTION_TO_TAG_ID['STOP'], 'STOP', confidence, contour)

        if aspect >= 1.4 and band_ratio > edge_band_ratio + 0.08 and circularity > 0.55:
            confidence = self._clip(0.35 + 0.45 * color_ratio + 0.20 * band_ratio, 0.0, 1.0)
            if confidence >= self.min_confidence:
                return self._build_detection(self.ACTION_TO_TAG_ID['STOP'], 'NO_ENTRY', confidence, contour)

        return None

    def _detect_blue_sign(self, img: np.ndarray, hsv: np.ndarray) -> Optional[DetectedSign]:
        # Lower saturation/value bounds are slightly relaxed to keep blue signs detectable
        # under dim lighting while shape checks still prevent generic false positives.
        lower = np.array([88, 45, 40], dtype=np.uint8)
        upper = np.array([136, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contour = self._largest_contour(mask)
        if contour is None:
            return None

        area, fill_ratio, color_ratio = self._score_contour(contour, mask)
        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(w) / float(h) if h else 0.0
        roi = img[y:y + h, x:x + w]
        gray = self._to_gray(roi)
        dark_mask = (gray < 75).astype(np.uint8)

        center_x = max(1, w // 2)
        center_y = max(1, h // 2)
        top_band = dark_mask[:max(1, h // 4), :]
        mid_col = dark_mask[:, max(0, center_x - max(1, w // 10)):center_x + max(1, w // 10)]
        dark_coords = np.column_stack(np.where(dark_mask > 0))

        dark_bbox_w = 0
        dark_bbox_h = 0
        centroid_x = center_x
        centroid_y = center_y
        if dark_coords.size:
            ys = dark_coords[:, 0]
            xs = dark_coords[:, 1]
            dark_bbox_w = int(xs.max() - xs.min() + 1)
            dark_bbox_h = int(ys.max() - ys.min() + 1)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())

        top_ratio = float(np.mean(top_band)) if top_band.size else 0.0
        mid_ratio = float(np.mean(mid_col)) if mid_col.size else 0.0
        horizontal_bias = (centroid_x - center_x) / max(1.0, center_x)
        vertical_bias = (centroid_y - center_y) / max(1.0, center_y)

        # Dead-end sign: square-ish blue background with a top bar and a vertical stem.
        if vertical_bias < -0.10 and top_ratio > 0.10 and mid_ratio > 0.12 and abs(horizontal_bias) < 0.18 and dark_bbox_w >= dark_bbox_h * 0.8:
            confidence = self._clip(0.30 + 0.40 * color_ratio + 0.30 * top_ratio, 0.0, 1.0)
            if confidence >= self.min_confidence:
                return self._build_detection(self.ACTION_TO_TAG_ID['DEAD_END'], 'DEAD_END', confidence, contour)

        # Tunnel: blue frame with a dark centered opening occupying much of the vertical area.
        opening_ratio = float(np.mean(dark_mask[:, :])) if dark_mask.size else 0.0
        opening_center_ratio = float(np.mean(dark_mask[:, max(0, center_x - max(1, w // 4)):center_x + max(1, w // 4)])) if dark_mask.size else 0.0
        opening_edge_ratio = max(
            float(np.mean(dark_mask[:, :max(1, w // 6)])) if dark_mask.size else 0.0,
            float(np.mean(dark_mask[:, -max(1, w // 6):])) if dark_mask.size else 0.0,
        )
        if aspect >= 1.1 and opening_center_ratio > 0.55 and opening_ratio > 0.35 and opening_center_ratio > opening_edge_ratio + 0.10:
            confidence = self._clip(0.30 + 0.35 * color_ratio + 0.35 * opening_center_ratio, 0.0, 1.0)
            if confidence >= self.min_confidence:
                return self._build_detection(self.ACTION_TO_TAG_ID['TUNNEL'], 'TUNNEL', confidence, contour)

        # Bridge: dark pillars near lower left/right with a clearer lower center lane.
        lower_band = dark_mask[max(0, int(h * 0.62)):, :]
        if lower_band.size:
            third = max(1, w // 3)
            left_low = lower_band[:, :third]
            center_low = lower_band[:, third:2 * third]
            right_low = lower_band[:, 2 * third:]
            left_ratio = float(np.mean(left_low)) if left_low.size else 0.0
            center_ratio = float(np.mean(center_low)) if center_low.size else 0.0
            right_ratio = float(np.mean(right_low)) if right_low.size else 0.0
            arch_hint = float(np.mean(dark_mask[max(0, int(h * 0.22)):max(1, int(h * 0.40)), third:2 * third]))

            if (
                left_ratio > 0.35 and
                right_ratio > 0.35 and
                center_ratio < 0.22 and
                arch_hint > 0.20 and
                abs(horizontal_bias) < 0.20
            ):
                confidence = self._clip(
                    0.25 + 0.30 * color_ratio + 0.20 * min(left_ratio, right_ratio) + 0.25 * (1.0 - center_ratio),
                    0.0,
                    1.0,
                )
                if confidence >= self.min_confidence:
                    return self._build_detection(self.ACTION_TO_TAG_ID['BRIDGE'], 'BRIDGE', confidence, contour)

        # Forward arrow: dark structure predominantly vertical and centered.
        if dark_bbox_h >= dark_bbox_w and vertical_bias < -0.05 and abs(horizontal_bias) < 0.28:
            confidence = self._clip(0.30 + 0.35 * fill_ratio + 0.35 * (1.0 - abs(horizontal_bias)), 0.0, 1.0)
            if confidence >= self.min_confidence:
                return self._build_detection(self.ACTION_TO_TAG_ID['FORWARD'], 'FORWARD', confidence, contour)

        # Horizontal arrow: direction by centroid bias.
        if dark_bbox_w > 0 and dark_bbox_w >= dark_bbox_h * 0.8:
            if horizontal_bias < -0.03:
                confidence = self._clip(0.30 + 0.35 * fill_ratio + 0.35 * abs(horizontal_bias), 0.0, 1.0)
                if confidence >= self.min_confidence:
                    return self._build_detection(self.ACTION_TO_TAG_ID['TURN_RIGHT'], 'TURN_RIGHT', confidence, contour)
            elif horizontal_bias > 0.03:
                confidence = self._clip(0.30 + 0.35 * fill_ratio + 0.35 * abs(horizontal_bias), 0.0, 1.0)
                if confidence >= self.min_confidence:
                    return self._build_detection(self.ACTION_TO_TAG_ID['TURN_LEFT'], 'TURN_LEFT', confidence, contour)

        return None

    def detect(self, img_arr: np.ndarray) -> Optional[DetectedSign]:
        if img_arr is None:
            return None

        img = img_arr.copy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if self.image_is_rgb else cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB if self.image_is_rgb else cv2.COLOR_RGBA2BGR)

        hsv = self._to_hsv(img)

        red = self._detect_red_sign(img, hsv)
        blue = self._detect_blue_sign(img, hsv)

        candidates = [det for det in (red, blue) if det is not None]
        if not candidates:
            return None

        best = max(candidates, key=lambda det: det.confidence)
        if self.debug:
            print(f"[OPENCV SIGN] label={best.label} tag_id={best.tag_id} conf={best.confidence:.2f}")

        return best