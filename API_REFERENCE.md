# Video Processing API Reference

## Overview

The Tracker API supports two video processing modes:

1. **Direct streaming** (small files <1 MB) → immediate real-time progress updates
2. **Chunked streaming** (large files >1 MB) → overlapped upload + processing

Both modes return Server-Sent Events (SSE) for real-time progress UI updates.

## Endpoints

### POST `/api/upload-chunk`

Upload a single chunk of a large video file. Chunks are processed as they arrive.

**Parameters (form data):**
- `session_id` (string, required): Unique session identifier (UUID format)
- `chunk_index` (integer, required): 0-based chunk number
- `total_chunks` (integer, required): Total number of chunks
- `ext` (string, required): File extension (e.g., `.mp4`, `.webm`, `.mkv`)
- `file` (file, required): Binary chunk data (max 1 MB)
- `confidence` (float, optional, default=0.25): YOLO detection threshold (0.0-1.0)
- `debug` (boolean, optional, default=false): Enable debug logging
- `preview_interval` (integer, optional, default=10): SSE event frequency (every N frames)

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_received": 3,
  "assembled": false,
  "streaming": true
}
```

**Error Responses:**
- `400`: Invalid session_id or extension format
- `413`: Chunk exceeds size limit

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('session_id', sessionId);
formData.append('chunk_index', 0);
formData.append('total_chunks', 10);
formData.append('ext', '.mp4');
formData.append('file', chunk);
formData.append('confidence', 0.25);
formData.append('preview_interval', 10);

const res = await fetch('/api/upload-chunk', {
  method: 'POST',
  body: formData,
});
const data = await res.json();
console.log('Chunks received:', data.chunks_received);
```

---

### GET `/api/stream-progress/{session_id}`

Stream Server-Sent Events with real-time processing progress for a chunked upload session. **Must be called after first chunk arrives** (or concurrently, with polling delay).

**Parameters (URL path):**
- `session_id` (string): Session identifier from `upload-chunk`

**Returns:** Server-Sent Events stream (HTTP 200, `Content-Type: text/event-stream`)

**Event Types:**

#### 1. `progress` Event

Sent every N frames (where N = `preview_interval`).

```
event: progress
data: {
  "frame_index": 42,
  "total_frames": 1000,
  "percent": 4.2,
  "preview": "base64_encoded_jpeg_data"
}
```

Fields:
- `frame_index` (integer): Current frame number (0-indexed)
- `total_frames` (integer | null): Total frames (null until metadata resolved)
- `percent` (float | null): Percentage complete (null until total_frames known)
- `preview` (string): Base64-encoded JPEG thumbnail (640px max width)

#### 2. `complete` Event

Sent once when processing finishes successfully.

```
event: complete
data: {
  "download_url": "/uploads/results/abc123def456.mp4",
  "fps": 30.0,
  "total_frames": 1000,
  "duration_seconds": 33.33,
  "width": 1920,
  "height": 1080,
  "detection_summary": {
    "person": 150,
    "dumbbell": 45,
    "barbell": 20
  },
  "frames_with_pose": 120,
  "processing_time_seconds": 180.5
}
```

Fields:
- `download_url` (string): Relative URL to download annotated MP4
- `fps` (float): Frames per second
- `total_frames` (integer): Total frames processed
- `duration_seconds` (float): Video duration in seconds
- `width` (integer): Frame width in pixels
- `height` (integer): Frame height in pixels
- `detection_summary` (object): Count of detected objects by label
- `frames_with_pose` (integer): Frames where pose was detected
- `processing_time_seconds` (float): Total processing duration

#### 3. `error` Event

Sent if an error occurs during processing.

```
event: error
data: {
  "detail": "FFmpeg decoder failed (broken pipe)"
}
```

#### 4. Keepalive Comments

Sent every 30 seconds if no progress event due to slow processing.

```
: keepalive
```

**Error Responses:**
- `400`: Invalid session_id format
- `404` (via error event): Session not found or timed out

**Example (JavaScript):**
```javascript
async function readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const eventBlock of events) {
      if (!eventBlock.trim()) continue;

      let eventType = 'message';
      let dataStr = '';
      for (const line of eventBlock.split('\n')) {
        if (line.startsWith('event: ')) eventType = line.slice(7).trim();
        else if (line.startsWith('data: ')) dataStr = line.slice(6);
      }

      if (!dataStr) continue;
      const data = JSON.parse(dataStr);

      if (eventType === 'progress') {
        console.log(`Frame ${data.frame_index}/${data.total_frames} (${data.percent}%)`);
      } else if (eventType === 'complete') {
        console.log('Processing finished!', data.download_url);
        return data;
      } else if (eventType === 'error') {
        throw new Error(data.detail);
      }
    }
  }
}

const response = await fetch(`/api/stream-progress/${sessionId}`);
const result = await readSSE(response);
```

---

### POST `/api/analyze-stream`

Upload a complete video file (<1 MB) and stream progress events in a single request. **Use for small files or as a fallback.**

**Parameters (form data):**
- `file` (file, required): Video file (MP4, WebM, MKV, AVI, MOV)
- `confidence` (float, optional, default=0.25): YOLO detection threshold
- `debug` (boolean, optional, default=false): Enable debug logging
- `preview_interval` (integer, optional, default=10): SSE event frequency

**Returns:** Server-Sent Events stream (same event types as `/stream-progress`)

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('file', videoFile);
formData.append('confidence', 0.25);
formData.append('preview_interval', 10);

const response = await fetch('/api/analyze-stream', {
  method: 'POST',
  body: formData,
});
const result = await readSSE(response);
```

---

### POST `/api/analyze-assembled/{session_id}`

Analyze a chunked upload session after all chunks are assembled. **Fallback endpoint** (automatically used if `/stream-progress` is unavailable).

**Parameters:**
- `session_id` (string): Session identifier
- `confidence` (float, optional, default=0.25): YOLO detection threshold
- `debug` (boolean, optional, default=false): Enable debug logging
- `preview_interval` (integer, optional, default=10): SSE event frequency

**Returns:** Server-Sent Events stream (same event types as `/stream-progress`)

---

## Complete Workflow: Large File Upload

```javascript
const sessionId = crypto.randomUUID();
const CHUNK_SIZE = 1 * 1024 * 1024; // 1 MB
const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

// 1. Upload all chunks
for (let i = 0; i < totalChunks; i++) {
  const start = i * CHUNK_SIZE;
  const chunk = file.slice(start, start + CHUNK_SIZE);

  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('chunk_index', i);
  formData.append('total_chunks', totalChunks);
  formData.append('ext', '.' + file.name.split('.').pop().toLowerCase());
  formData.append('file', chunk);
  // confidence, debug, preview_interval on first chunk or all chunks

  const uploadRes = await fetch('/api/upload-chunk', {
    method: 'POST',
    body: formData,
  });

  if (!uploadRes.ok) {
    const error = await uploadRes.json();
    throw new Error(error.detail);
  }

  console.log(`Uploaded chunk ${i + 1}/${totalChunks}`);
}

// 2. Connect to progress stream (processing already started)
const sseRes = await fetch(`/api/stream-progress/${sessionId}`);

if (!sseRes.ok) {
  // Fallback to analyze-assembled
  const fallbackRes = await fetch(`/api/analyze-assembled/${sessionId}`, {
    method: 'POST',
  });
  const result = await readSSE(fallbackRes);
} else {
  const result = await readSSE(sseRes);
}

// 3. Download result
window.location.href = result.download_url;
```

---

## Supported Video Formats

| Format | Extension | Container | Notes |
|--------|-----------|-----------|-------|
| MP4    | `.mp4`    | H.264/H.265 | Most compatible; may have moov at end |
| WebM   | `.webm`   | VP8/VP9   | Good for streaming (moov at start) |
| MKV    | `.mkv`    | Matroska  | Supports any codec; metadata flexible |
| MOV    | `.mov`    | QuickTime | Apple standard; often moov at start |
| AVI    | `.avi`    | RIFF      | Legacy; reliable but larger file size |

---

## Error Handling

### Network Interruption Mid-Upload

If the browser connection drops during chunk upload:

1. Client-side: catch fetch error, retry chunk or give up
2. Server-side: session remains in memory; reaper cleans up after 10 minutes

### FFmpeg Decode Failure

If FFmpeg encounters a corrupt frame:

1. Error event sent via SSE
2. Processing stops; `complete` event not sent
3. Session cleaned up

### Timeout

If no SSE events received for 35 seconds:

1. Keepalive comment sent (prevents proxy timeout)
2. Client-side: handle empty lines gracefully (most SSE parsers do)

---

## Configuration

**Frontend** (`frontend/src/App.jsx`):
```javascript
const CHUNK_SIZE = 1 * 1024 * 1024;      // Split files > 1 MB
const CHUNK_TIMEOUT_MS = 30_000;          // Abort chunk upload after 30 seconds
```

**Backend** (`backend/app/routers/analysis.py`):
```python
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
```

**Session cleanup** (`backend/app/main.py`):
```python
SESSION_TIMEOUT = 600  # 10 minutes (stale sessions cleaned up)
```

---

## Performance Tips

1. **Adjust `preview_interval`:**
   - Higher value (50+): fewer SSE events, less network overhead
   - Lower value (5): more frequent updates, smoother progress UI

2. **Confidence threshold:**
   - `0.25` (default): catches more detections, slower processing
   - `0.5`: balanced
   - `0.75+`: only high-confidence detections, faster

3. **YOLO model size:**
   - Consider using `yolov8n` (nano, ~6 MB) for real-time processing
   - Larger models (`m`, `l`) have better accuracy but slower inference

---

## Status Endpoint

### GET `/api/health`

Quick health check.

**Response:**
```json
{"status": "ok"}
```
