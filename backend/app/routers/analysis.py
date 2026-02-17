# Analysis router — accepts a video upload, processes every frame with
# detection + pose estimation, renders overlays, and returns an annotated MP4.

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

import json

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import VideoAnalysisResponse
from app.services.detection import DetectionService
from app.services.pose import PoseService
from app.services.rendering import render_overlays
from app.services.video import VideoProcessor, encode_frame_preview

logger = logging.getLogger(__name__)
router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile,
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold (0-1)"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    """Upload a gym/weightlifting video and get an annotated MP4 with overlays."""

    # Validate by file extension (content_type is unreliable for video)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Use MP4, AVI, MOV, WebM, or MKV.",
        )

    # Stream upload to temp file (handles large files without loading into RAM)
    tmp_fd, tmp_input_path = tempfile.mkstemp(suffix=ext)
    try:
        with open(tmp_fd, "wb") as tmp_in:
            while chunk := await file.read(8 * 1024 * 1024):  # 8 MB chunks
                tmp_in.write(chunk)

        start_time = time.time()

        # Initialise video processor
        result_id = uuid.uuid4().hex[:12]
        output_filename = f"{result_id}.mp4"
        output_path = str(RESULTS_DIR / output_filename)
        processor = VideoProcessor(tmp_input_path, output_path)
        metadata = processor.get_metadata()

        logger.info(
            "Processing video: %d frames, %.1f fps, %dx%d, %.1fs",
            metadata["total_frames"], metadata["fps"],
            metadata["width"], metadata["height"],
            metadata["duration_seconds"],
        )

        # Initialise ML services
        detector = get_detection_service()
        poser = get_pose_service()

        # Process every frame
        detection_counts: dict[str, int] = {}
        frames_with_pose = 0

        for frame_idx, frame_bgr in processor.frames():
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Detection
            detections = detector.detect(
                frame_bgr,
                confidence_threshold=confidence,
                debug=(debug and frame_idx == 0),  # debug-log first frame only
            )

            # Aggregate detection counts
            for det in detections:
                label = det["label"]
                detection_counts[label] = detection_counts.get(label, 0) + 1

            # Pose estimation
            pose_data = poser.estimate(frame_rgb)
            if pose_data is not None:
                frames_with_pose += 1

            # Render overlays in-place (frame is a fresh decode, safe to mutate)
            annotated = render_overlays(frame_bgr, detections, pose_data, inplace=True)

            # Write annotated frame to output
            processor.write_frame(annotated)

            # Log progress periodically
            if frame_idx % 100 == 0 and frame_idx > 0:
                logger.info("Processed %d / %d frames", frame_idx, metadata["total_frames"])

        # Finalise output video (re-encode to H.264)
        processor.finalize()

        elapsed = time.time() - start_time
        logger.info("Video processing complete in %.1fs", elapsed)

        return VideoAnalysisResponse(
            download_url=f"/uploads/results/{output_filename}",
            fps=metadata["fps"],
            total_frames=metadata["total_frames"],
            duration_seconds=metadata["duration_seconds"],
            width=metadata["width"],
            height=metadata["height"],
            detection_summary=detection_counts,
            frames_with_pose=frames_with_pose,
            processing_time_seconds=round(elapsed, 2),
        )
    finally:
        # Clean up temp input file
        Path(tmp_input_path).unlink(missing_ok=True)


def _sse_event(event: str, data: dict) -> str:
    """Format a server-sent event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/analyze-stream")
async def analyze_video_stream(
    file: UploadFile,
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold (0-1)"),
    debug: bool = Query(False, description="Enable debug logging"),
    preview_interval: int = Query(10, ge=1, le=100, description="Send preview every N frames"),
):
    """Upload a video and stream annotated frame previews via SSE during processing."""

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Use MP4, AVI, MOV, WebM, or MKV.",
        )

    tmp_fd, tmp_input_path = tempfile.mkstemp(suffix=ext)
    try:
        with open(tmp_fd, "wb") as tmp_in:
            while chunk := await file.read(8 * 1024 * 1024):
                tmp_in.write(chunk)
    except Exception:
        Path(tmp_input_path).unlink(missing_ok=True)
        raise

    def generate():
        processor = None
        try:
            start_time = time.time()

            result_id = uuid.uuid4().hex[:12]
            output_filename = f"{result_id}.mp4"
            output_path = str(RESULTS_DIR / output_filename)
            processor = VideoProcessor(tmp_input_path, output_path)
            metadata = processor.get_metadata()

            logger.info(
                "SSE processing video: %d frames, %.1f fps, %dx%d",
                metadata["total_frames"], metadata["fps"],
                metadata["width"], metadata["height"],
            )

            detector = get_detection_service()
            poser = get_pose_service()

            detection_counts: dict[str, int] = {}
            frames_with_pose = 0
            total_frames = metadata["total_frames"]
            last_event_time = time.time()

            for frame_idx, frame_bgr in processor.frames():
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                detections = detector.detect(
                    frame_bgr,
                    confidence_threshold=confidence,
                    debug=(debug and frame_idx == 0),
                )

                for det in detections:
                    label = det["label"]
                    detection_counts[label] = detection_counts.get(label, 0) + 1

                pose_data = poser.estimate(frame_rgb)
                if pose_data is not None:
                    frames_with_pose += 1

                annotated = render_overlays(frame_bgr, detections, pose_data, inplace=True)
                processor.write_frame(annotated)

                if frame_idx % preview_interval == 0 or frame_idx == total_frames - 1:
                    preview_b64 = encode_frame_preview(annotated)
                    yield _sse_event("progress", {
                        "frame_index": frame_idx,
                        "total_frames": total_frames,
                        "percent": round(frame_idx / max(total_frames, 1) * 100, 1),
                        "preview": preview_b64,
                    })
                    last_event_time = time.time()
                elif time.time() - last_event_time > 30:
                    # Keepalive comment to prevent proxy/browser from closing idle SSE connection
                    yield ": keepalive\n\n"
                    last_event_time = time.time()

            processor.finalize()

            elapsed = time.time() - start_time
            logger.info("SSE video processing complete in %.1fs", elapsed)

            yield _sse_event("complete", {
                "download_url": f"/uploads/results/{output_filename}",
                "fps": metadata["fps"],
                "total_frames": metadata["total_frames"],
                "duration_seconds": metadata["duration_seconds"],
                "width": metadata["width"],
                "height": metadata["height"],
                "detection_summary": detection_counts,
                "frames_with_pose": frames_with_pose,
                "processing_time_seconds": round(elapsed, 2),
            })

        except Exception as exc:
            logger.exception("Error during SSE video processing")
            yield _sse_event("error", {"detail": str(exc)})

        finally:
            Path(tmp_input_path).unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
