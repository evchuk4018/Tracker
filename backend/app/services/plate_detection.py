# CV-based weight plate detector — complements YOLO by detecting the
# distinctive circular/disc shape of barbell plates using OpenCV.

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class CVPlateDetector:
    """Detect weight plates using Hough circles and contour/ellipse fitting.

    Uses strict filtering (gradient direction verification, high edge
    continuity requirements) to minimise false positives in complex scenes.
    """

    def __init__(
        self,
        min_radius_frac: float = 0.03,
        max_radius_frac: float = 0.15,
        hough_dp: float = 1.2,
        hough_param1: float = 100,
        hough_param2: float = 60,
        min_circularity: float = 0.82,
        min_ellipse_axis_ratio: float = 0.5,
        blur_ksize: int = 9,
        max_detections: int = 6,
    ) -> None:
        self.min_radius_frac = min_radius_frac
        self.max_radius_frac = max_radius_frac
        self.hough_dp = hough_dp
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.min_circularity = min_circularity
        self.min_ellipse_axis_ratio = min_ellipse_axis_ratio
        self.blur_ksize = blur_ksize
        self.max_detections = max_detections

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run both detection strategies, merge, and return plate detections."""
        h, w = image.shape[:2]
        min_dim = min(h, w)
        min_r = max(int(min_dim * self.min_radius_frac), 15)
        max_r = int(min_dim * self.max_radius_frac)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 2)
        edges = cv2.Canny(blurred, 50, int(self.hough_param1))

        # Precompute gradient fields for direction verification
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        hough_dets = self._detect_hough_circles(
            gray, blurred, edges, grad_x, grad_y, min_r, max_r, h, w,
        )
        contour_dets = self._detect_ellipse_contours(
            gray, edges, grad_x, grad_y, min_r, max_r, h, w,
        )

        return self._deduplicate(hough_dets + contour_dets)[:self.max_detections]

    # ------------------------------------------------------------------
    # Strategy 1: Hough Circle Transform (front-facing plates)
    # ------------------------------------------------------------------

    def _detect_hough_circles(
        self,
        gray: np.ndarray,
        blurred: np.ndarray,
        edges: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        min_r: int,
        max_r: int,
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=max(min_r * 2, h // 15),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        candidates: list[dict[str, Any]] = []
        if circles is None:
            return candidates

        for cx, cy, r in np.round(circles[0]).astype(int):
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)

            conf = self._score_candidate(
                gray, edges, grad_x, grad_y, cx, cy, r, r, h, w,
            )
            if conf < 0.45:
                continue

            candidates.append({
                "label": "weight_plate",
                "confidence": round(conf, 3),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "proxy": False,
            })

        return candidates

    # ------------------------------------------------------------------
    # Strategy 2: Contour + ellipse fitting (angled/side views)
    # ------------------------------------------------------------------

    def _detect_ellipse_contours(
        self,
        gray: np.ndarray,
        edges: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        min_r: int,
        max_r: int,
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[dict[str, Any]] = []
        min_area = math.pi * min_r * min_r * 0.5
        max_area = math.pi * max_r * max_r * 1.5

        for contour in contours:
            if len(contour) < 5:
                continue

            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis_w, axis_h), _angle = ellipse

            minor = min(axis_w, axis_h)
            major = max(axis_w, axis_h)
            if major == 0:
                continue
            axis_ratio = minor / major
            if axis_ratio < self.min_ellipse_axis_ratio:
                continue

            major_r = major / 2
            if major_r < min_r or major_r > max_r:
                continue

            rx, ry = axis_w / 2, axis_h / 2
            conf = self._score_candidate(
                gray, edges, grad_x, grad_y, cx, cy, rx, ry, h, w,
            )
            conf *= (0.5 + 0.5 * circularity) * (0.5 + 0.5 * axis_ratio)
            if conf < 0.45:
                continue

            rect = cv2.boundingRect(contour)
            bx, by, bw, bh = rect

            candidates.append({
                "label": "weight_plate",
                "confidence": round(min(conf, 0.95), 3),
                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                "proxy": False,
            })

        return candidates

    # ------------------------------------------------------------------
    # Scoring: edge continuity + gradient direction verification
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        gray: np.ndarray,
        edges: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        h: int,
        w: int,
    ) -> float:
        n_samples = 36
        margin = 2
        edge_hits = 0
        radial_hits = 0
        total_samples = 0

        for i in range(n_samples):
            angle = 2 * math.pi * i / n_samples
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            px = int(cx + rx * cos_a)
            py = int(cy + ry * sin_a)

            if not (0 <= px < w and 0 <= py < h):
                continue

            total_samples += 1

            # Check edge presence with tight margin
            y_lo = max(0, py - margin)
            y_hi = min(h, py + margin + 1)
            x_lo = max(0, px - margin)
            x_hi = min(w, px + margin + 1)
            if edges[y_lo:y_hi, x_lo:x_hi].any():
                edge_hits += 1

            # Gradient direction check: gradient at the perimeter should
            # point radially (toward or away from center).
            gx = grad_x[py, px]
            gy = grad_y[py, px]
            grad_mag = math.sqrt(gx * gx + gy * gy)
            if grad_mag > 10:  # only check where gradient is meaningful
                # Normalised dot product between gradient and radial direction
                dot = (gx * cos_a + gy * sin_a) / grad_mag
                # Accept both inward and outward gradients
                if abs(dot) > 0.5:
                    radial_hits += 1

        if total_samples == 0:
            return 0.0

        edge_ratio = edge_hits / total_samples
        radial_ratio = radial_hits / total_samples

        # Require strong edge continuity
        if edge_ratio < 0.55:
            return 0.0

        # Require meaningful gradient alignment
        if radial_ratio < 0.3:
            return 0.0

        # Combine edge continuity and gradient alignment into confidence
        base_confidence = (edge_ratio * 0.6 + radial_ratio * 0.4)
        base_confidence = min(base_confidence, 0.95)

        # Bonus for center hole (characteristic of weight plates)
        inner_edge_hits = 0
        inner_samples = 0
        inner_rx, inner_ry = rx * 0.3, ry * 0.3
        for i in range(n_samples):
            angle = 2 * math.pi * i / n_samples
            px = int(cx + inner_rx * math.cos(angle))
            py = int(cy + inner_ry * math.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                inner_samples += 1
                y_lo = max(0, py - margin)
                y_hi = min(h, py + margin + 1)
                x_lo = max(0, px - margin)
                x_hi = min(w, px + margin + 1)
                if edges[y_lo:y_hi, x_lo:x_hi].any():
                    inner_edge_hits += 1

        if inner_samples > 0 and inner_edge_hits > inner_samples * 0.25:
            base_confidence = min(base_confidence + 0.1, 0.95)

        return base_confidence

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self, candidates: list[dict[str, Any]], iou_threshold: float = 0.4,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        sorted_cands = sorted(
            candidates, key=lambda d: d["confidence"], reverse=True,
        )
        kept: list[dict[str, Any]] = []

        for cand in sorted_cands:
            overlaps = any(
                compute_iou(cand["bbox"], existing["bbox"]) > iou_threshold
                for existing in kept
            )
            if not overlaps:
                kept.append(cand)

        return kept


class ColorPlateDetector:
    """Detect weight plates by HSV color masking.

    Much more stable frame-to-frame than shape detection alone — the user
    picks a pixel from the first frame and we track that specific hue through
    the whole video.
    """

    def __init__(
        self,
        r: int,
        g: int,
        b: int,
        hue_tol: int = 18,
        sat_floor: int = 40,
        val_floor: int = 30,
        max_detections: int = 8,
    ) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.hue_tol = hue_tol
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.max_detections = max_detections

        # Precompute HSV range(s) from the sample RGB.
        # OpenCV uses BGR order, so wrap in a 1×1 pixel image.
        sample_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
        sample_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
        h_center = int(sample_hsv[0, 0, 0])
        s_center = int(sample_hsv[0, 0, 1])
        v_center = int(sample_hsv[0, 0, 2])

        logger.info(
            "[ColorPlateDetector] input RGB=(%d,%d,%d) -> HSV center=(%d,%d,%d)",
            r, g, b, h_center, s_center, v_center,
        )

        # Saturation / value floors respect the sampled values so very dark or
        # desaturated plates still get a sensible range.
        s_lo = max(30, min(s_center - 50, sat_floor))
        v_lo = max(20, min(v_center - 60, val_floor))

        h_lo = h_center - hue_tol
        h_hi = h_center + hue_tol

        # Red hue wraps around 0/179 in OpenCV HSV.
        if h_lo < 0:
            self._ranges = [
                (np.array([0, s_lo, v_lo]), np.array([h_hi, 255, 255])),
                (np.array([180 + h_lo, s_lo, v_lo]), np.array([179, 255, 255])),
            ]
        elif h_hi > 179:
            self._ranges = [
                (np.array([h_lo, s_lo, v_lo]), np.array([179, 255, 255])),
                (np.array([0, s_lo, v_lo]), np.array([h_hi - 180, 255, 255])),
            ]
        else:
            self._ranges = [
                (np.array([h_lo, s_lo, v_lo]), np.array([h_hi, 255, 255])),
            ]

        # Near-grayscale (black/white/gray plates): ignore hue entirely.
        if s_center < 50:
            self._ranges = [
                (np.array([0, 0, v_lo]), np.array([179, 60, 255])),
            ]

        logger.info(
            "[ColorPlateDetector] HSV ranges: %s",
            [(lo.tolist(), hi.tolist()) for lo, hi in self._ranges],
        )

        # Morphological kernels.
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return weight_plate detections based on color masking."""
        h, w = image.shape[:2]
        min_dim = min(h, w)
        min_r = max(15, int(min_dim * 0.03))
        max_r = int(min_dim * 0.18)
        min_area = math.pi * min_r * min_r * 0.4
        max_area = math.pi * max_r * max_r * 1.6

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Build combined mask from all hue ranges.
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self._ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        # Morphological cleanup: close fills gaps inside the plate disc,
        # open removes small noise blobs.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * math.pi * area) / (perimeter * perimeter)
            if circularity < 0.55:
                continue

            # Convexity guard: reject crescent / arc shapes.
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < 0.70:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            conf = round(min(circularity * 1.1, 0.95), 3)

            candidates.append({
                "label": "weight_plate",
                "confidence": conf,
                "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                "proxy": False,
            })

        # Deduplicate and cap.
        candidates = _deduplicate(candidates)
        return candidates[: self.max_detections]


def _deduplicate(
    candidates: list[dict[str, Any]], iou_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Shared IoU deduplication helper."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in sorted_cands:
        if not any(
            compute_iou(cand["bbox"], ex["bbox"]) > iou_threshold for ex in kept
        ):
            kept.append(cand)
    return kept
