# Detection service — wraps a YOLO model for object detection.
# Swapping models: replace the `model_path` or subclass DetectionService.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Available pre-trained models
AVAILABLE_MODELS = {
    "coco": "yolov8n.pt",           # COCO dataset (80 classes, includes person)
    "oiv7": "yolov8m-oiv7.pt",      # Open Images V7 (600 classes, includes dumbbell)
    "coco-medium": "yolov8m.pt",    # Larger COCO model for better accuracy
    "coco-large": "yolov8l.pt",     # Even larger COCO model
}

# Class remapping for gym equipment terminology
# Maps model classes to user-friendly gym equipment names
GYM_EQUIPMENT_REMAP = {
    # --- Direct OIV7 gym classes ---
    "Dumbbell": "dumbbell",
    "Training bench": "bench",
    "Treadmill": "treadmill",
    "Indoor rower": "rowing_machine",
    "Stationary bicycle": "stationary_bike",
    "Sports equipment": "gym_equipment",

    # --- Weight plate proxies (OIV7 classes visually similar to plates) ---
    "Plate": "weight_plate",           # dinner plate shape ≈ weight plate
    "Tire": "weight_plate",            # rubber round ≈ bumper plate
    "Wheel": "weight_plate",           # round with center hole
    "Flying disc": "weight_plate",     # disc/frisbee shape
    "Ball": "weight_plate",            # generic round object
    "Bicycle wheel": "weight_plate",   # round with center hub

    # --- Barbell proxies ---
    "Horizontal bar": "barbell",       # gymnastics bar ≈ barbell
    "Baseball bat": "barbell",         # rod shape (fallback)
}

# OIV7 classes used as proxies for gym equipment — accept at lower confidence
EQUIPMENT_PROXY_CLASSES = {
    "Plate", "Tire", "Wheel", "Flying disc", "Ball",
    "Bicycle wheel", "Horizontal bar", "Baseball bat",
}

# Confidence floor for proxy classes
PROXY_CONFIDENCE_THRESHOLD = 0.15

LABEL_MAP: dict[int, str] = {
    0: "person",
    # The following are non-standard; a fine-tuned model would populate them.
    # We keep them here so swapping to a custom-trained model is seamless.
}


class DetectionService:
    """Object detection using YOLOv8."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize detection service with specified model.

        Args:
            model_path: Path to model file or model key from AVAILABLE_MODELS.
                       If None, uses YOLO_MODEL env var or defaults to 'oiv7'.
        """
        if model_path is None:
            # Check environment variable first
            model_key = os.getenv("YOLO_MODEL", "oiv7")
            model_path = AVAILABLE_MODELS.get(model_key, model_key)
        elif model_path in AVAILABLE_MODELS:
            # If it's a known key, get the actual path
            model_path = AVAILABLE_MODELS[model_path]

        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.35,
        return_all_classes: bool = False,
        debug: bool = False,
    ) -> list[dict[str, Any]]:
        """Run detection and return list of detections.

        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence (0-1) to include detection
            return_all_classes: Include class_id in results
            debug: Enable verbose logging of all detections

        Each detection dict:
            label: str
            confidence: float
            bbox: [x1, y1, x2, y2]  (pixel coords)
            class_id: int (optional, if return_all_classes or debug is True)
        """
        results = self.model(image, verbose=False)[0]
        detections: list[dict[str, Any]] = []
        filtered_count = 0

        if debug:
            logger.info(f"Total boxes detected by YOLO: {len(results.boxes)}")

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_label = results.names.get(cls_id, f"class_{cls_id}")

            # Apply gym equipment label remapping
            label = GYM_EQUIPMENT_REMAP.get(raw_label, raw_label)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if debug:
                logger.info(f"  Class {cls_id:2d} ({raw_label:20s} → {label:20s}): confidence={conf:.3f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

            # Proxy classes use a lower confidence floor
            if raw_label in EQUIPMENT_PROXY_CLASSES:
                effective_threshold = min(confidence_threshold, PROXY_CONFIDENCE_THRESHOLD)
            else:
                effective_threshold = confidence_threshold

            if conf < effective_threshold:
                filtered_count += 1
                continue

            is_proxy = raw_label in EQUIPMENT_PROXY_CLASSES

            detection = {
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                "proxy": is_proxy,
            }

            if return_all_classes or debug:
                detection["class_id"] = cls_id
                detection["raw_label"] = raw_label

            detections.append(detection)

        if debug:
            logger.info(f"Filtered out {filtered_count} detections below threshold {confidence_threshold}")
            logger.info(f"Returning {len(detections)} detections")

        return detections
