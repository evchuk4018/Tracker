# Detection service — wraps a YOLO model for object detection.
# Swapping models: replace the `model_path` or subclass DetectionService.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs we care about for gym context
# We'll detect everything but highlight barbell-related objects specially.
# COCO doesn't have "weight plate" or "barbell" natively so we rely on
# fine-tuning or fall back to "sports ball" (32) and general objects.
# With stock YOLO we detect *person* (0) and any other objects present.

LABEL_MAP: dict[int, str] = {
    0: "person",
    # The following are non-standard; a fine-tuned model would populate them.
    # We keep them here so swapping to a custom-trained model is seamless.
}


class DetectionService:
    """Object detection using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Run detection and return list of detections.

        Each detection dict:
            label: str
            confidence: float
            bbox: [x1, y1, x2, y2]  (pixel coords)
        """
        results = self.model(image, verbose=False)[0]
        detections: list[dict[str, Any]] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            label = results.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                }
            )

        return detections
