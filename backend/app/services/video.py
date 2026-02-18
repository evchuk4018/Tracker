# Video I/O service — reads input video frames, writes annotated output MP4.

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
import threading
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


class StreamingDecoder:
    """Decodes video by piping raw bytes to FFmpeg stdin, reading PPM frames from stdout.

    This enables processing to start as soon as chunk data arrives, rather than
    waiting for the full file to be assembled on disk.
    """

    def __init__(self, output_path: str) -> None:
        self._output_path = output_path

        ffmpeg_exe = _get_ffmpeg_exe()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg is required for streaming decode")
        self._ffmpeg_exe = ffmpeg_exe

        self._proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "image2pipe",
                "-c:v", "ppm",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Drain stderr in background to prevent FFmpeg blocking if buffer fills
        self._stderr_data = b""
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_thread.start()

        # Output writer — created lazily on first decoded frame (dimensions unknown until then)
        self._intermediate_path = tempfile.mktemp(suffix=".avi")
        self._writer: cv2.VideoWriter | None = None
        self._width: int | None = None
        self._height: int | None = None

        # Metadata — set later via set_metadata() once assembled file is probed
        self._fps: float = 30.0
        self._total_frames: int | None = None
        self._duration: float | None = None

    # -- stdin feeding ----------------------------------------------------------

    def write_input(self, data: bytes) -> None:
        """Write raw video-file bytes to FFmpeg stdin."""
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def close_input(self) -> None:
        """Signal end-of-input to FFmpeg."""
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass

    # -- stdout frame reading ---------------------------------------------------

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly *n* bytes from *stream*, or return None on EOF."""
        data = b""
        while len(data) < n:
            chunk = stream.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield decoded BGR frames as they become available from FFmpeg."""
        stdout = self._proc.stdout
        while True:
            # PPM header: "P6\n"
            magic = stdout.readline()
            if not magic:
                return  # EOF — FFmpeg finished
            if magic.strip() != b"P6":
                continue  # skip unexpected data

            # Dimensions line (skip comment lines starting with '#')
            while True:
                line = stdout.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"#"):
                    break

            parts = line.split()
            if len(parts) != 2:
                continue
            width, height = int(parts[0]), int(parts[1])

            # Max-value line (always "255")
            maxval = stdout.readline()
            if not maxval:
                return

            # Pixel data: width * height * 3 bytes (RGB)
            nbytes = width * height * 3
            pixel_data = self._read_exact(stdout, nbytes)
            if pixel_data is None:
                return  # unexpected EOF

            frame_rgb = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                height, width, 3,
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Lazy-init VideoWriter on first frame
            if self._writer is None:
                self._width = width
                self._height = height
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._writer = cv2.VideoWriter(
                    self._intermediate_path, fourcc, 30.0,
                    (width, height),
                )
                if not self._writer.isOpened():
                    raise ValueError("Cannot create video writer for streaming output")

            yield frame_bgr

    # -- output writing ---------------------------------------------------------

    def write_output_frame(self, frame_bgr: np.ndarray) -> None:
        """Write one annotated frame to the intermediate output video."""
        if self._writer is not None:
            self._writer.write(frame_bgr)

    # -- metadata ---------------------------------------------------------------

    def set_metadata(
        self, fps: float, total_frames: int, width: int, height: int, duration: float,
    ) -> None:
        """Store metadata obtained from probing the assembled file."""
        self._fps = fps
        self._total_frames = total_frames
        if self._width is None:
            self._width = width
        if self._height is None:
            self._height = height
        self._duration = duration

    # -- finalize ---------------------------------------------------------------

    def finalize(self) -> str:
        """Release resources and re-encode intermediate AVI to H.264 MP4."""
        if self._writer is not None:
            self._writer.release()

        # Wait for FFmpeg decode process to finish
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

        if self._proc.returncode != 0:
            logger.error(
                "FFmpeg streaming decode exited with code %d: %s",
                self._proc.returncode, self._stderr_data.decode(errors="replace"),
            )

        # Re-encode intermediate AVI → H.264 MP4
        fps = self._fps or 30.0
        cmd = [
            self._ffmpeg_exe, "-y",
            "-r", str(fps),
            "-i", self._intermediate_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self._output_path,
        ]
        logger.info("Re-encoding streaming output to H.264: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("ffmpeg re-encode failed: %s", result.stderr)
            Path(self._intermediate_path).rename(self._output_path)
        else:
            Path(self._intermediate_path).unlink(missing_ok=True)

        return self._output_path

    # -- cleanup / helpers ------------------------------------------------------

    def cleanup(self) -> None:
        """Kill FFmpeg process and remove intermediate files."""
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        Path(self._intermediate_path).unlink(missing_ok=True)

    def _drain_stderr(self) -> None:
        """Read all stderr output from FFmpeg (prevents pipe buffer deadlock)."""
        try:
            self._stderr_data = self._proc.stderr.read()
        except Exception:
            pass
        finally:
            try:
                self._proc.stderr.close()
            except Exception:
                pass


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
