# Weightlifting Scene Analyzer

Full-stack application that analyzes weightlifting videos and images using computer vision. Upload a gym video or image and get:

- **Object detection** (YOLOv8) — bounding boxes with labels and confidence scores for detected objects (person, sports equipment, etc.)
- **Pose estimation** (MediaPipe) — stick-figure skeleton overlay on detected lifters
- **Annotated output** — composite video/image with all overlays rendered

**Video features:**
- Chunked streaming uploads (optimized for files >1 MB)
- Real-time processing progress with live frame previews via Server-Sent Events
- Automatic MP4 output with H.264 encoding
- Supports MP4, WebM, MKV, MOV, AVI formats

## Architecture

```
backend/
  app/
    main.py              # FastAPI entry point + session reaper
    routers/
      analysis.py        # /api/analyze*, /api/upload-chunk, /api/stream-progress
    services/
      detection.py       # YOLO object detection (swappable)
      pose.py            # MediaPipe pose estimation (swappable)
      rendering.py       # OpenCV overlay rendering
      video.py           # Video I/O (VideoProcessor, StreamingDecoder)
    models/
      schemas.py         # Pydantic request/response models
  uploads/
    chunks/              # Temporary chunked video segments
    results/             # Annotated result files
  requirements.txt

frontend/
  src/
    App.jsx              # Main UI — upload video/image, progress, results
    App.css              # Styles
    index.css            # Global styles
    main.jsx             # React entry
  vite.config.js         # Dev server with API proxy
```

**Key components:**

- **StreamingDecoder**: Pipes raw video bytes to FFmpeg stdin, reads decoded PPM frames from stdout for immediate processing
- **ProcessingSession**: Manages dual-thread pipeline (chunk writer + ML processor) with SSE event streaming
- **SessionReaper**: Background task cleans up abandoned sessions after 10 minutes

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads YOLOv8n (~6 MB) and MediaPipe pose models automatically.

### Frontend

```bash
cd frontend
npm install --include=dev
npm run dev
```

Opens on `http://localhost:5173`. API calls proxy to the backend at `:8000`.

## API

**See [API_REFERENCE.md](API_REFERENCE.md) for complete documentation.**

### Video Processing

#### POST `/api/analyze-stream` (small files <1 MB)

Upload a complete video and stream progress events.

```bash
curl -X POST -F "file=@video.mp4" \
  -F "confidence=0.25" \
  -F "preview_interval=10" \
  http://localhost:8000/api/analyze-stream
```

Returns SSE stream with `progress` and `complete` events.

#### POST `/api/upload-chunk` (large files >1 MB)

Upload video in 1 MB chunks. Processing starts on chunk 0 arrival.

```bash
# Chunk 0
curl -X POST \
  -F "session_id=<uuid>" \
  -F "chunk_index=0" \
  -F "total_chunks=10" \
  -F "ext=.mp4" \
  -F "file=@chunk_0" \
  http://localhost:8000/api/upload-chunk

# ... upload chunks 1-9 ...
```

#### GET `/api/stream-progress/{session_id}` (SSE)

Stream real-time processing progress for a chunked upload.

```javascript
const eventSource = new EventSource(`/api/stream-progress/${sessionId}`);
eventSource.addEventListener('progress', (event) => {
  const data = JSON.parse(event.data);
  console.log(`Frame ${data.frame_index}/${data.total_frames}`);
  // Display preview thumbnail: <img src="data:image/jpeg;base64,${data.preview}" />
});
eventSource.addEventListener('complete', (event) => {
  const result = JSON.parse(event.data);
  console.log(`Download: ${result.download_url}`);
  eventSource.close();
});
```

### Image Processing (Legacy)

#### POST `/api/analyze`

Upload an image (JPEG/PNG/WebP, max 20 MB) as multipart form data.

**Response:**
```json
{
  "detections": [
    {"label": "person", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
  ],
  "pose": {
    "keypoints": [{"id": 0, "name": "nose", "x": 320, "y": 150, "z": -0.1, "visibility": 0.95}],
    "connections": [[11, 12], [11, 13]]
  },
  "annotated_image_url": "/uploads/abc123.jpg"
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

## Swapping Models

**Detection:** Subclass or replace `DetectionService` in `backend/app/services/detection.py`. Any model that takes an image array and returns `[{label, confidence, bbox}]` works.

**Pose:** Subclass or replace `PoseService` in `backend/app/services/pose.py`. Must return `{keypoints, connections}` or `None`.

**Custom-trained YOLO:** Train on a dataset with weight plate / barbell classes, then pass the `.pt` path:
```python
DetectionService(model_path="path/to/custom_weights.pt")
```

## Video Streaming Architecture

Large video files (>1 MB) are processed using a **streaming pipeline**:

1. **Frontend chunks file** into 1 MB segments and uploads them sequentially
2. **Backend starts processing** as soon as chunk 0 arrives (no waiting for all chunks)
3. **FFmpeg pipes** raw bytes to stdout as decoded frames (PPM format)
4. **ML inference** (YOLO + MediaPipe) runs frame-by-frame immediately
5. **SSE stream** pushes progress events and live frame previews to frontend
6. **Final output** is re-encoded to H.264 MP4 for efficient delivery

**Benefits:**
- Parallelized upload + processing (processes while uploading)
- Real-time progress feedback with live frame previews
- Automatic metadata resolution after assembly
- Graceful handling of all video formats (moov-at-start and moov-at-end MP4, WebM, MKV, etc.)
- Session cleanup for abandoned uploads

**See [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) for technical deep dive.**

## Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete API endpoint documentation with examples
- **[STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md)** — Technical design, implementation details, performance characteristics
- **[SUMMARY.md](SUMMARY.md)** — Project overview and features
