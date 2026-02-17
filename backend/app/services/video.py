# Video I/O service — reads input video frames, writes annotated output MP4.

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _get_ffmpeg_exe() -> str | None:
    """Return path to ffmpeg binary, or None if unavailable."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoProcessor:
    """Handles video I/O: reading frames from input, writing annotated output."""

    def __init__(self, input_path: str, output_path: str) -> None:
        self._input_path = input_path
        self._output_path = output_path

        self._cap = cv2.VideoCapture(input_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Write to intermediate AVI with MJPG (universally reliable in OpenCV)
        self._intermediate_path = tempfile.mktemp(suffix=".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(
            self._intermediate_path, fourcc, self._fps,
            (self._width, self._height),
        )
        if not self._writer.isOpened():
            raise ValueError("Cannot create video writer for output.")

    def get_metadata(self) -> dict:
        """Return video metadata."""
        duration = self._total_frames / self._fps if self._fps > 0 else 0.0
        return {
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
            "total_frames": self._total_frames,
            "duration_seconds": round(duration, 2),
        }

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_bgr) for every frame in the video."""
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """Write a single annotated frame to the output video."""
        self._writer.write(frame_bgr)

    def finalize(self) -> str:
        """Release resources and re-encode to H.264 MP4. Returns output path."""
        self._cap.release()
        self._writer.release()

        ffmpeg = _get_ffmpeg_exe()
        if ffmpeg:
            cmd = [
                ffmpeg, "-y",
                "-i", self._intermediate_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                self._output_path,
            ]
            logger.info("Re-encoding to H.264: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.error("ffmpeg failed: %s", result.stderr)
                # Fall back to the intermediate AVI
                Path(self._intermediate_path).rename(self._output_path)
            else:
                Path(self._intermediate_path).unlink(missing_ok=True)
        else:
            # No ffmpeg available — use the AVI directly
            logger.warning("ffmpeg not found, using MJPG AVI as output")
            Path(self._intermediate_path).rename(self._output_path)

        return self._output_path


def encode_frame_preview(
    frame_bgr: np.ndarray,
    max_width: int = 640,
    jpeg_quality: int = 70,
) -> str:
    """Resize frame to thumbnail, JPEG-encode, and return base64 string."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not success:
        raise ValueError("Failed to JPEG-encode frame")

    return base64.b64encode(buffer).decode("ascii")
