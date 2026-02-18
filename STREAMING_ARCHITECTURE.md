# Streaming Video Decode Architecture

## Overview

The Tracker application now supports **streaming video processing** where ML inference (object detection and pose estimation) begins as soon as the first chunk of a large video file arrives, rather than waiting for the entire file to be uploaded and assembled.

This overlapping of upload + decode + processing time significantly reduces total latency for multi-megabyte video files.

## Problem Statement

**Before:** Upload → Assemble → Process
```
Upload all chunks (1-2 min) → Assemble on disk (seconds) → ML processing (varies)
Total latency = upload + assembly + processing
```

**After:** Upload (parallel with processing)
```
Upload chunk 0 → Processing starts immediately (parallel with remaining uploads)
Total latency = max(remaining upload time, processing time)
```

### Challenge: Incomplete Video Files

MP4 and other video containers often store metadata (`moov` atom) at the **end** of the file. A partially-received file cannot be opened by OpenCV (`cv2.VideoCapture`), breaking the simple approach of opening the file as chunks arrive.

**Solution:** Pipe raw bytes directly to FFmpeg, which handles its own buffering and media parsing. FFmpeg outputs self-delimiting frames (PPM format) that can be processed immediately without waiting for the complete file.

## Architecture

### Data Flow

```
┌─────────────────────────────────────────┐
│         Frontend (React)                 │
│                                         │
│  1. Upload chunks (parallel-safe loop) │
│  2. POST /api/upload-chunk            │
│  3. GET /api/stream-progress (SSE)    │
└──────────────┬──────────────────────────┘
               │
               │ HTTP POST (1 MB chunks)
               ▼
┌──────────────────────────────────────────────────────┐
│         Backend (FastAPI + ProcessingSession)        │
│                                                      │
│  ┌───────────────────────────────────────────┐      │
│  │         ProcessingSession (in-memory)     │      │
│  │                                           │      │
│  │  input_queue ──┐                         │      │
│  │                │                         │      │
│  │                ▼                         │      │
│  │    _writer_thread                        │      │
│  │    (reads queue → writes FFmpeg stdin)  │      │
│  │                │                         │      │
│  │                ▼                         │      │
│  │  ┌─────────────────────────────────┐   │      │
│  │  │  FFmpeg subprocess               │   │      │
│  │  │  (decodes from stdin)            │   │      │
│  │  │  (outputs PPM to stdout)         │   │      │
│  │  └─────────────────────────────────┘   │      │
│  │                │                         │      │
│  │                ▼                         │      │
│  │    _processor_thread                     │      │
│  │    (reads PPM frames → YOLO → MediaPipe)│      │
│  │    (renders overlays → writes AVI)      │      │
│  │                │                         │      │
│  │                ▼                         │      │
│  │  event_queue (SSE events)               │      │
│  └───────────────────────────────────────────┘      │
│                                                      │
│  /api/stream-progress/{session_id}                  │
│  (polls event_queue, returns SSE stream)            │
└──────────────────────────────────────────────────────┘
               ▲
               │ SSE (text/event-stream)
               │
         Frontend updates UI
```

### Components

#### 1. `StreamingDecoder` (backend/app/services/video.py)

Manages the FFmpeg subprocess and frame pipeline.

```python
class StreamingDecoder:
    def __init__(self, output_path: str):
        # Start FFmpeg with:
        # Input: stdin (raw video bytes)
        # Output: stdout (PPM-encoded frames)
        # Stderr: captured in background thread (prevent deadlock)

    def write_input(self, data: bytes):
        # Write chunk bytes to FFmpeg stdin

    def close_input(self):
        # Signal EOF to FFmpeg

    def frames() -> Generator[np.ndarray]:
        # Parse PPM headers and yield decoded BGR frames
        # PPM format: P6\n{width} {height}\n255\n{RGB pixels}

    def write_output_frame(self, frame_bgr: np.ndarray):
        # Write annotated frame to intermediate AVI

    def set_metadata(fps, total_frames, width, height, duration):
        # Store metadata from assembled file probe

    def finalize() -> str:
        # Re-encode intermediate AVI to H.264 MP4 with correct framerate
```

#### 2. `ProcessingSession` (backend/app/routers/analysis.py)

Orchestrates the dual-thread pipeline and SSE event streaming.

```python
class ProcessingSession:
    def __init__(self, session_id, ext, total_chunks, confidence, debug, preview_interval):
        # Create StreamingDecoder
        # Create input_queue and event_queue
        # Start _writer_thread and _processor_thread

    def feed_chunk(self, data: bytes):
        # Called by upload_chunk endpoint
        # Adds chunk to input_queue
        # Sends sentinel when last chunk arrives

    def resolve_metadata(self, assembled_path: Path):
        # Called after assembly (when all chunks received)
        # Opens assembled file, extracts fps/total_frames/dimensions
        # Calls decoder.set_metadata()

    def _writer_loop():
        # Daemon thread
        # Reads from input_queue, writes to FFmpeg stdin
        # Exits when sentinel received

    def _processor_loop():
        # Daemon thread
        # Reads PPM frames from FFmpeg stdout
        # Runs YOLO detection + MediaPipe pose on each frame
        # Renders overlays
        # Writes annotated frame to intermediate AVI
        # Pushes SSE progress events to event_queue

    def cleanup():
        # Release FFmpeg, delete intermediate files
        # Remove session from registry
```

#### 3. Modified `/api/upload-chunk` Endpoint

```python
@router.post("/upload-chunk")
async def upload_chunk(
    session_id, chunk_index, total_chunks, ext, file,
    confidence: float = 0.25,     # ML threshold
    debug: bool = False,
    preview_interval: int = 10,   # SSE event frequency
):
    # 1. Read chunk into memory (1 MB max, safe)
    # 2. Save to disk (still needed for metadata extraction)
    # 3. On chunk_index == 0: Create ProcessingSession, register globally
    # 4. Feed chunk to session (starts writer_thread)
    # 5. When all chunks received: Assemble file, call resolve_metadata()
    # 6. Return {"streaming": True, ...}
```

#### 4. New `/api/stream-progress/{session_id}` Endpoint (SSE)

```python
@router.get("/api/stream-progress/{session_id}")
async def stream_progress(session_id: str):
    # Returns StreamingResponse immediately (no async blocking in handler)
    # Generator function (runs in threadpool worker) polls for session:
    #   - Waits up to 30 seconds for session creation
    #   - Pulls SSE events from event_queue with 35-second timeout
    #   - Yields "progress" and "complete" events
    #   - Cleans up session on final event

    # Event types:
    # - "progress": {frame_index, total_frames, percent, preview}
    #   (percent = null until metadata resolved)
    # - "keepalive": empty (prevents proxy timeout on slow streams)
    # - "complete": full result (fps, duration, detection_summary, etc.)
    # - "error": {detail: "error message"}
```

#### 5. Session Reaper (backend/app/main.py)

Background asyncio task cleans up abandoned sessions.

```python
@app.on_event("startup")
async def _start_session_reaper():
    # Every 60 seconds, check for sessions older than 10 minutes
    # Call cleanup() on stale sessions
```

### Frontend Flow

#### Old Flow (Sequential Assembly)

```
1. Upload all chunks
   for i in 0..totalChunks-1:
       POST /api/upload-chunk
2. All chunks received
   POST /api/analyze-assembled/{sessionId}  // SSE stream
   // Processing happens now
```

**Problem:** Codespaces proxy returns 502 when two long-lived HTTP connections (SSE + chunk uploads) are open concurrently.

#### New Flow (Sequential Upload → SSE)

```javascript
// 1. Upload all chunks (server starts processing on chunk 0)
for (let i = 0; i < totalChunks; i++) {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('chunk_index', i);
    formData.append('total_chunks', totalChunks);
    formData.append('ext', ext);
    formData.append('file', chunk);
    // + confidence, debug, preview_interval on chunk 0

    const uploadRes = await fetch('/api/upload-chunk', {
        method: 'POST',
        body: formData,
    });

    setProgress({ phase: 'uploading', chunk: i+1, totalChunks, percent });
}

// 2. All uploads done — connect to SSE (processing already running)
setProgress({ phase: 'processing', processing: null });

const sseRes = await fetch(`/api/stream-progress/${sessionId}`);
const sseResult = await readSSE(sseRes);  // Polls SSE events
setResult(sseResult);
```

**Benefit:** Upload and processing are **not** concurrent (avoids 502), but processing still starts during the upload loop because the server begins on chunk 0 arrival.

#### Progress State Shape

```javascript
// During upload phase:
{ phase: 'uploading', chunk: i+1, totalChunks, percent: 15 }

// After upload, during processing:
{ phase: 'processing', processing: { frame_index: 42, total_frames: 1000, percent: 4.2, preview: "base64..." } }

// Before metadata resolved (total_frames unknown):
{ phase: 'processing', processing: { frame_index: 42, total_frames: null, percent: null, preview: "base64..." } }
```

## Frame Processing Pipeline

### 1. Writer Thread

```
input_queue (bytes)
    │
    ├─ chunk 0 (0 bytes → 1 MB)
    ├─ chunk 1 (1 MB → 2 MB)
    ├─ chunk 2 (2 MB → 3 MB)
    └─ None (sentinel)
    │
    ▼
FFmpeg stdin (raw video file bytes)
    For most formats: processing starts immediately
    For moov-at-end MP4: FFmpeg buffers until last chunk arrives
```

### 2. FFmpeg Decoding

```
FFmpeg -i pipe:0 -f image2pipe -c:v ppm pipe:1
    │
    Input: raw MP4/WebM/MKV bytes from stdin
    │
    Processing: decode video frames as data arrives
    │
    Output: PPM-encoded frames on stdout
```

PPM (Portable Pixmap) format is self-delimiting:
```
P6                    # Magic number
{width} {height}      # Dimensions (parsed to know frame size)
255                   # Max color value
{RGB pixel bytes}     # Exactly width*height*3 bytes (no separators)
[next frame]
```

### 3. Processor Thread

```
for frame_bgr in decoder.frames():
    1. Convert BGR → RGB
    2. Run YOLO detection (with confidence threshold)
    3. Run MediaPipe pose estimation
    4. Render overlays (bounding boxes, pose skeleton)
    5. Write annotated frame to intermediate AVI (MJPG codec)
    6. If frame_idx % preview_interval == 0:
       - Encode frame to JPEG thumbnail
       - Push SSE progress event with base64 preview, frame_index, total_frames, percent
```

### 4. Output Finalization

After all frames processed:

```
1. Release intermediate AVI writer and FFmpeg process
2. Probe assembled file (if not already done) for metadata
3. Re-encode intermediate AVI → H.264 MP4 with correct framerate
   (preserves high-quality processing, final compression for delivery)
4. Push "complete" event with download URL and statistics
```

## Metadata Timing

A key challenge: **video dimensions and frame count are unknown until streaming completes** (especially for moov-at-end formats).

### Solution: Two-Phase Metadata

**Phase 1: Before assembly complete**
- `total_frames = None`, `percent = None` in SSE events
- UI shows "Processing frame 42..." without progress bar

**Phase 2: After assembly complete**
- Upload handler calls `session.resolve_metadata(assembled_path)`
- Opens assembled file with `cv2.VideoCapture`, extracts metadata
- Calls `decoder.set_metadata()`
- Subsequent SSE events include `total_frames` and `percent`

For formats with moov-at-start (MP4 with correct atomization, WebM, MKV), metadata may be extracted immediately without waiting for assembly.

## Fallback Paths

### No SSE Support

If `/api/stream-progress` returns non-200:
```javascript
// Fall back to legacy endpoint
const fallbackRes = await fetch(`/api/analyze-assembled/${sessionId}`, { method: 'POST' });
const fallbackResult = await readSSE(fallbackRes);
```

This endpoint exists unchanged and still works.

### Small Files (<1 MB)

```javascript
// Direct upload, no chunking
POST /api/analyze-stream
// Returns SSE stream (existing code path)
```

## Configuration

### Frontend

`frontend/src/App.jsx`:
```javascript
const CHUNK_SIZE = 1 * 1024 * 1024;      // 1 MB
const CHUNK_TIMEOUT_MS = 30_000;          // 30 seconds per chunk
const PREVIEW_INTERVAL = 10;              // SSE event every 10 frames
```

### Backend

`backend/app/routers/analysis.py`:
```python
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

# Session reaper timeout
SESSION_TIMEOUT = 600  # 10 minutes (in main.py)
```

## Monitoring & Debugging

### Logs

```
# When session created
logger.info("Streaming session %s created", session_id)

# While processing
logger.info("Streaming session %s complete in %.1fs (%d frames)", session_id, elapsed, frame_idx)

# Errors
logger.exception("Processor thread error for session %s", session_id)
```

### Frontend Console

```javascript
// Progress events printed to console during development:
setProgress({ phase: 'uploading', chunk: 5, totalChunks: 10, percent: 50 })
setProgress({ phase: 'processing', processing: { frame_index: 120, total_frames: 1000, percent: 12.0 } })
```

## Performance Characteristics

### Assumptions
- 1 GB video file → 1000 chunks × 1 MB each
- Each chunk uploads in ~500 ms (typical network speed)
- Full video takes 10 minutes to process

### Timeline

**Sequential (old approach):**
- Upload: 500 ms × 1000 = 500 seconds (8+ minutes)
- Assembly: ~5 seconds
- ML processing: 600 seconds (10 minutes)
- **Total: ~618 seconds (10+ minutes)**

**Streaming (new approach):**
- Chunk 0 arrives at t=500 ms → processing starts immediately
- Processing runs parallel with remaining uploads
- Remaining chunks: 500 s × 999 = ~500 seconds
- Processing: 600 seconds (starts at t=500ms, finishes at t=600.5s)
- **Total: max(500 + 0.5, 600) ≈ 600 seconds (10 minutes)**
- Plus first chunk overhead, but significant wins for slower processors or faster networks

For faster networks (higher chunk arrival rate) or slower ML models, the overlapping effect saves even more time.

## Testing

### Manual Testing

1. **Small file (<1 MB):**
   - Upload directly via `/api/analyze-stream`
   - Verify live progress previews

2. **Large file (>1 MB):**
   - Frontend splits into chunks
   - Monitor `/uploads/chunks/{sessionId}/` directory on server
   - Verify `stream-progress` SSE events arrive with `frame_index`, `total_frames` (null initially), `percent` (null initially), `preview`
   - Verify metadata resolves and `percent` becomes non-null after assembly

3. **Network interruption:**
   - Kill browser mid-upload
   - Verify session removed after 10 minutes by reaper

4. **Corrupt video:**
   - Upload truncated MP4
   - Verify FFmpeg error forwarded via SSE

### Automated Testing

Unit tests for:
- `StreamingDecoder.frames()` with mock FFmpeg subprocess
- `ProcessingSession` event queueing
- `upload_chunk` with multiple chunks
- Metadata resolution timing

Integration tests:
- End-to-end large file upload + processing
- SSE stream parsing at frontend
- Fallback to legacy endpoint

## Limitations & Future Work

1. **FFmpeg required** — `imageio_ffmpeg` must be available. Backend gracefully falls back to `VideoProcessor` for single-file uploads if FFmpeg unavailable.

2. **Memory usage** — intermediate AVI written to disk before H.264 re-encoding. For very large videos, disk I/O may bottleneck.

3. **Concurrent session limit** — no explicit limit on number of active sessions. Server resources determine practical limit (typically 2-3 concurrent 1080p streams on modern hardware).

4. **Metadata before processing** — ideal would be to extract metadata immediately on container (MP4) header arrival, avoiding the assembly step. Requires more sophisticated MP4 parsing.

5. **Real-time processing** — frame processing is serialized (YOLO → MediaPipe → render per frame). Could parallelize YOLO on future frames while current frame undergoes pose estimation.

## References

- **FFmpeg PPM output:** https://ffmpeg.org/ffmpeg-formats.html#ppm
- **OpenCV VideoWriter:** https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html
- **FastAPI SSE:** https://fastapi.tiangolo.com/advanced/sse/
- **Server-Sent Events (MDN):** https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
