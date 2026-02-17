# Analysis router — single endpoint that accepts an image upload,
# runs detection + pose estimation, renders overlays, and returns results.

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, Query

from app.models.schemas import AnalysisResponse
from app.services.detection import DetectionService
from app.services.pose import PoseService
from app.services.rendering import render_overlays

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Module-level singletons (initialised lazily)
_detection_service: DetectionService | None = None
_pose_service: PoseService | None = None


def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


def get_pose_service() -> PoseService:
    global _pose_service
    if _pose_service is None:
        _pose_service = PoseService()
    return _pose_service


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile,
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold (0-1)"),
    debug: bool = Query(False, description="Enable debug logging"),
    show_all_classes: bool = Query(False, description="Include class IDs in response"),
):
    """Upload a gym/weightlifting image and get annotated results.

    Debug Parameters:
    - confidence: Lower this to see more detections (try 0.1-0.2)
    - debug: Set to true to see detailed logging in server console
    - show_all_classes: Include COCO class IDs in detection results
    """

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 20 MB).")

    # Decode image
    nparr = np.frombuffer(raw_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- Run detection ---
    detector = get_detection_service()
    detections = detector.detect(
        image_bgr,
        confidence_threshold=confidence,
        return_all_classes=show_all_classes,
        debug=debug,
    )

    # --- Run pose estimation ---
    poser = get_pose_service()
    pose_data = poser.estimate(image_rgb)

    # --- Render overlays ---
    annotated_bgr = render_overlays(image_bgr, detections, pose_data)

    # Save annotated image
    result_id = uuid.uuid4().hex[:12]
    out_filename = f"{result_id}.jpg"
    out_path = UPLOAD_DIR / out_filename
    cv2.imwrite(str(out_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return AnalysisResponse(
        detections=detections,
        pose=pose_data,
        annotated_image_url=f"/uploads/{out_filename}",
    )
