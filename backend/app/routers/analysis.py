# Analysis router — accepts a video upload, processes every frame with
# detection + pose estimation, renders overlays, and returns an annotated MP4.

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import json
import re
import shutil

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import VideoAnalysisResponse
from app.services.detection import DetectionService
from app.services.pose import PoseService
from app.services.rendering import render_overlays
from app.services.video import VideoProcessor, StreamingDecoder, encode_frame_preview

logger = logging.getLogger(__name__)
router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9\-]{1,64}$')

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

# ---------------------------------------------------------------------------
# Streaming processing sessions — overlaps chunk upload with ML processing
# ---------------------------------------------------------------------------
_active_sessions: dict[str, ProcessingSession] = {}
_sessions_lock = threading.Lock()


class ProcessingSession:
    """Manages the lifecycle of a streaming video processing session.

    Created on the first chunk upload.  A writer thread feeds chunk data to
    FFmpeg's stdin while a processor thread reads decoded frames and runs
    detection + pose estimation.  SSE events are pushed to ``event_queue``
    for the ``stream-progress`` endpoint to relay to the browser.
    """

    def __init__(
        self,
        session_id: str,
        ext: str,
        total_chunks: int,
        confidence: float,
        debug: bool,
        preview_interval: int,
    ) -> None:
        self.session_id = session_id
        self.ext = ext
        self.total_chunks = total_chunks
        self.chunks_received = 0
        self.confidence = confidence
        self.debug = debug
        self.preview_interval = preview_interval
        self.start_time = time.time()

        self.session_dir = CHUNKS_DIR / session_id

        result_id = uuid.uuid4().hex[:12]
        self.output_filename = f"{result_id}.mp4"
        self.output_path = str(RESULTS_DIR / self.output_filename)

        self.decoder = StreamingDecoder(self.output_path)

        # Unbounded — chunks are 1 MB each, safe for memory
        self.input_queue: queue.Queue[bytes | None] = queue.Queue()
        self.event_queue: queue.Queue[dict] = queue.Queue()

        # Metadata placeholders (resolved after assembly)
        self._fps: float | None = None
        self._total_frames: int | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._duration: float | None = None

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"writer-{session_id[:8]}",
            daemon=True,
        )
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name=f"processor-{session_id[:8]}",
            daemon=True,
        )
        self._writer_thread.start()
        self._processor_thread.start()

    # -- chunk feeding ----------------------------------------------------------

    def feed_chunk(self, data: bytes) -> None:
        """Add chunk data to the input queue (called from upload handler)."""
        self.input_queue.put(data)
        self.chunks_received += 1
        if self.chunks_received == self.total_chunks:
            self.input_queue.put(None)  # sentinel — no more data

    def resolve_metadata(self, assembled_path: Path) -> None:
        """Extract video metadata from the assembled file on disk."""
        cap = cv2.VideoCapture(str(assembled_path))
        if cap.isOpened():
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._duration = (
                self._total_frames / self._fps if self._fps > 0 else 0.0
            )
            cap.release()
            self.decoder.set_metadata(
                self._fps, self._total_frames,
                self._width, self._height, self._duration,
            )

    # -- background threads -----------------------------------------------------

    def _writer_loop(self) -> None:
        """Feed chunk data from input_queue to FFmpeg stdin."""
        try:
            while True:
                data = self.input_queue.get()
                if data is None:
                    break
                self.decoder.write_input(data)
            self.decoder.close_input()
        except BrokenPipeError:
            logger.error("FFmpeg stdin pipe broken for session %s", self.session_id)
            self.event_queue.put({
                "type": "error",
                "data": {"detail": "FFmpeg decoder failed (broken pipe)"},
            })
        except Exception as exc:
            logger.exception("Writer thread error for session %s", self.session_id)
            self.event_queue.put({
                "type": "error",
                "data": {"detail": str(exc)},
            })

    def _processor_loop(self) -> None:
        """Read decoded frames from FFmpeg, run ML, write output."""
        try:
            detector = get_detection_service()
            poser = get_pose_service()

            detection_counts: dict[str, int] = {}
            frames_with_pose = 0
            frame_idx = 0
            last_event_time = time.time()

            for frame_bgr in self.decoder.frames():
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                detections = detector.detect(
                    frame_bgr,
                    confidence_threshold=self.confidence,
                    debug=(self.debug and frame_idx == 0),
                )
                for det in detections:
                    label = det["label"]
                    detection_counts[label] = detection_counts.get(label, 0) + 1

                pose_data = poser.estimate(frame_rgb)
                if pose_data is not None:
                    frames_with_pose += 1

                annotated = render_overlays(
                    frame_bgr, detections, pose_data, inplace=True,
                )
                self.decoder.write_output_frame(annotated)

                if frame_idx % self.preview_interval == 0:
                    preview_b64 = encode_frame_preview(annotated)
                    total = self._total_frames
                    percent = (
                        round(frame_idx / total * 100, 1)
                        if total and total > 0
                        else None
                    )
                    self.event_queue.put({
                        "type": "progress",
                        "data": {
                            "frame_index": frame_idx,
                            "total_frames": total,
                            "percent": percent,
                            "preview": preview_b64,
                        },
                    })
                    last_event_time = time.time()
                elif time.time() - last_event_time > 30:
                    self.event_queue.put({"type": "keepalive", "data": None})
                    last_event_time = time.time()

                frame_idx += 1

            # Finalize output video
            self.decoder.finalize()

            elapsed = time.time() - self.start_time
            logger.info(
                "Streaming session %s complete in %.1fs (%d frames)",
                self.session_id, elapsed, frame_idx,
            )

            fps = self._fps or 30.0
            total_frames = self._total_frames or frame_idx
            width = self._width or (self.decoder._width or 0)
            height = self._height or (self.decoder._height or 0)
            duration = self._duration or (
                total_frames / fps if fps > 0 else 0.0
            )

            self.event_queue.put({
                "type": "complete",
                "data": {
                    "download_url": f"/uploads/results/{self.output_filename}",
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration_seconds": round(duration, 2),
                    "width": width,
                    "height": height,
                    "detection_summary": detection_counts,
                    "frames_with_pose": frames_with_pose,
                    "processing_time_seconds": round(elapsed, 2),
                },
            })

        except Exception as exc:
            logger.exception(
                "Processor thread error for session %s", self.session_id,
            )
            self.event_queue.put({
                "type": "error",
                "data": {"detail": str(exc)},
            })

    # -- cleanup ----------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources and remove session from registry."""
        self.decoder.cleanup()
        shutil.rmtree(self.session_dir, ignore_errors=True)
        with _sessions_lock:
            _active_sessions.pop(self.session_id, None)


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


@router.post("/upload-chunk")
async def upload_chunk(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    ext: str = Form(...),
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    debug: bool = Form(False),
    preview_interval: int = Form(10),
):
    """Receive one chunk of a large file upload.

    On the first chunk, a streaming processing pipeline is started so that
    decoding and ML inference overlap with the remaining uploads.
    """
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not re.match(r'^\.[a-zA-Z0-9]{1,10}$', ext):
        raise HTTPException(status_code=400, detail="Invalid extension")

    session_dir = CHUNKS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Read the full chunk into memory (max 1 MB — safe)
    chunk_data = await file.read()

    # Save to disk (still needed for assembly / metadata extraction)
    chunk_path = session_dir / f"{chunk_index:06d}.bin"
    with open(chunk_path, "wb") as f:
        f.write(chunk_data)

    received = len(list(session_dir.glob("*.bin")))
    assembled = received == total_chunks

    # --- Streaming processing integration ---
    with _sessions_lock:
        session = _active_sessions.get(session_id)

    if session is None and chunk_index == 0:
        session = ProcessingSession(
            session_id=session_id,
            ext=ext,
            total_chunks=total_chunks,
            confidence=confidence,
            debug=debug,
            preview_interval=preview_interval,
        )
        with _sessions_lock:
            _active_sessions[session_id] = session

    if session is not None:
        session.feed_chunk(chunk_data)

    # Assemble on disk when all chunks received (for metadata extraction)
    if assembled:
        assembled_path = session_dir / f"assembled{ext}"
        with open(assembled_path, "wb") as out:
            for i in range(total_chunks):
                cp = session_dir / f"{i:06d}.bin"
                out.write(cp.read_bytes())
                cp.unlink()
        if session is not None:
            session.resolve_metadata(assembled_path)

    return {
        "session_id": session_id,
        "chunks_received": received,
        "assembled": assembled,
        "streaming": session is not None,
    }


@router.post("/analyze-assembled/{session_id}")
async def analyze_assembled(
    session_id: str,
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    debug: bool = Query(False),
    preview_interval: int = Query(10, ge=1, le=100),
):
    """Run analysis on a previously assembled chunked upload, streaming progress via SSE."""
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    session_dir = CHUNKS_DIR / session_id
    assembled_files = list(session_dir.glob("assembled.*"))
    if not assembled_files:
        raise HTTPException(status_code=404, detail="Session not found or not yet assembled")

    tmp_input_path = str(assembled_files[0])

    def generate():
        try:
            start_time = time.time()

            result_id = uuid.uuid4().hex[:12]
            output_filename = f"{result_id}.mp4"
            output_path = str(RESULTS_DIR / output_filename)
            processor = VideoProcessor(tmp_input_path, output_path)
            metadata = processor.get_metadata()

            logger.info(
                "Assembled-session processing: %d frames, %.1f fps, %dx%d",
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
                    yield ": keepalive\n\n"
                    last_event_time = time.time()

            processor.finalize()

            elapsed = time.time() - start_time
            logger.info("Assembled-session processing complete in %.1fs", elapsed)

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
            logger.exception("Error during assembled-session processing")
            yield _sse_event("error", {"detail": str(exc)})

        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stream-progress/{session_id}")
async def stream_progress(session_id: str):
    """Stream SSE progress events for an active streaming processing session.

    The frontend opens this connection concurrently with chunk uploads.
    The generator polls for the session inside the response body (not in the
    handler) so that HTTP headers are sent immediately, avoiding a deadlock
    with the upload requests that create the session.
    """
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    def generate():
        # Poll for the session to be created (happens when chunk 0 arrives)
        session: ProcessingSession | None = None
        for _ in range(300):  # up to 30 seconds
            with _sessions_lock:
                session = _active_sessions.get(session_id)
            if session is not None:
                break
            time.sleep(0.1)

        if session is None:
            yield _sse_event("error", {"detail": "Session not found or creation timed out"})
            return

        try:
            while True:
                try:
                    event = session.event_queue.get(timeout=35)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue

                if event["type"] == "keepalive":
                    yield ": keepalive\n\n"
                    continue

                yield _sse_event(event["type"], event["data"])

                if event["type"] in ("complete", "error"):
                    break
        finally:
            session.cleanup()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
