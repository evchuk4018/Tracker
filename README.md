# Weightlifting Scene Analyzer

Full-stack application that analyzes weightlifting photos using computer vision. Upload a gym image and get:

- **Object detection** (YOLOv8) — bounding boxes with labels and confidence scores for detected objects (person, sports equipment, etc.)
- **Pose estimation** (MediaPipe) — stick-figure skeleton overlay on detected lifters
- **Annotated output** — composite image with all overlays rendered

## Architecture

```
backend/
  app/
    main.py              # FastAPI entry point
    routers/
      analysis.py        # /api/analyze endpoint
    services/
      detection.py       # YOLO object detection (swappable)
      pose.py            # MediaPipe pose estimation (swappable)
      rendering.py       # OpenCV overlay rendering
    models/
      schemas.py         # Pydantic request/response models
  uploads/               # Annotated result images
  requirements.txt

frontend/
  src/
    App.jsx              # Main UI — upload, preview, results
    App.css              # Styles
    index.css            # Global styles
    main.jsx             # React entry
  vite.config.js         # Dev server with API proxy
```

Each ML service (detection, pose) is a standalone class. To swap models, replace the class or change the model path — no other code changes needed.

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

### `POST /api/analyze`

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
