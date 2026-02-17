# Overlay renderer — draws bounding boxes and stick-figure skeleton on the image.

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Colour palette (BGR)
COLOR_BBOX = (0, 255, 0)           # green for general detections
COLOR_PERSON = (255, 200, 0)       # cyan-ish for person bbox
COLOR_SKELETON = (0, 140, 255)     # orange for skeleton lines
COLOR_JOINT = (0, 0, 255)          # red for joints
COLOR_TEXT_BG = (0, 0, 0)
COLOR_DUMBBELL = (0, 255, 255)     # yellow - for dumbbells
COLOR_BARBELL = (255, 0, 255)      # magenta - for barbells
COLOR_WEIGHT_PLATE = (0, 140, 255) # orange - for weight plates
COLOR_BENCH = (255, 128, 0)        # blue - for benches

# Map label names to colors (model-agnostic, works with any YOLO backend)
LABEL_COLORS = {
    "person": COLOR_PERSON,
    "dumbbell": COLOR_DUMBBELL,
    "barbell": COLOR_BARBELL,
    "weight_plate": COLOR_WEIGHT_PLATE,
    "bench": COLOR_BENCH,
    "gym_equipment": COLOR_BBOX,
    "treadmill": (128, 128, 0),
    "rowing_machine": (128, 128, 0),
    "stationary_bike": (128, 128, 0),
}


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    inplace: bool = False,
) -> np.ndarray:
    """Draw bounding boxes with labels and confidence scores."""
    img = image if inplace else image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]
        class_id = det.get("class_id")

        # Choose color based on label name
        color = LABEL_COLORS.get(label, COLOR_BBOX)

        # Proxy detections get thinner border to distinguish them
        is_proxy = det.get("proxy", False)
        thickness = 2 if is_proxy else 3

        # Enhanced label showing class ID when in debug mode
        if class_id is not None:
            text = f"{label} ({class_id}) {conf:.0%}"
        else:
            text = f"{label} {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label background and text
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def draw_skeleton(
    image: np.ndarray,
    pose_data: dict[str, Any],
    visibility_threshold: float = 0.5,
    inplace: bool = False,
) -> np.ndarray:
    """Draw stick-figure skeleton over the image."""
    img = image if inplace else image.copy()
    keypoints = pose_data["keypoints"]
    connections = pose_data["connections"]

    kp_map = {kp["id"]: kp for kp in keypoints}

    # Draw connections
    for id_a, id_b in connections:
        a, b = kp_map.get(id_a), kp_map.get(id_b)
        if a is None or b is None:
            continue
        if a["visibility"] < visibility_threshold or b["visibility"] < visibility_threshold:
            continue
        pt_a = (int(a["x"]), int(a["y"]))
        pt_b = (int(b["x"]), int(b["y"]))
        cv2.line(img, pt_a, pt_b, COLOR_SKELETON, 3, cv2.LINE_AA)

    # Draw joints
    for kp in keypoints:
        if kp["visibility"] < visibility_threshold:
            continue
        center = (int(kp["x"]), int(kp["y"]))
        cv2.circle(img, center, 5, COLOR_JOINT, -1, cv2.LINE_AA)
        cv2.circle(img, center, 5, COLOR_SKELETON, 1, cv2.LINE_AA)

    return img


def render_overlays(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    pose_data: dict[str, Any] | None,
    inplace: bool = False,
) -> np.ndarray:
    """Composite all overlays onto the image."""
    img = draw_detections(image, detections, inplace=inplace)
    if pose_data is not None:
        img = draw_skeleton(img, pose_data, inplace=True)
    return img
