import { useState, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = '';
const CHUNK_SIZE = 1 * 1024 * 1024; // 1 MB — small enough for Codespaces proxy
const CHUNK_TIMEOUT_MS = 30_000;     // 30 s per chunk before giving up

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function App() {
  const [file, setFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(null);
  const fileInputRef = useRef(null);
  const calibCanvasRef = useRef(null);
  const [calibFrame, setCalibFrame] = useState(null);
  const [plateColor, setPlateColor] = useState(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    const allowedTypes = [
      'video/mp4', 'video/quicktime', 'video/x-msvideo',
      'video/webm', 'video/x-matroska',
    ];
    const allowedExts = ['.mp4', '.mov', '.avi', '.webm', '.mkv'];
    const ext = '.' + f.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(f.type) && !allowedExts.includes(ext)) {
      setError('Please upload a video file (MP4, MOV, AVI, WebM, MKV).');
      return;
    }
    setFile(f);
    setVideoPreview(URL.createObjectURL(f));
    setError(null);
    setResult(null);
    setCalibFrame(null);
    setPlateColor(null);
    calibCanvasRef.current = null;

    // Extract first frame client-side for color calibration.
    const vid = document.createElement('video');
    const objUrl = URL.createObjectURL(f);
    vid.src = objUrl;
    vid.currentTime = 0.05;
    vid.onseeked = () => {
      const c = document.createElement('canvas');
      c.width = vid.videoWidth;
      c.height = vid.videoHeight;
      c.getContext('2d').drawImage(vid, 0, 0);
      setCalibFrame(c.toDataURL('image/jpeg', 0.85));
      calibCanvasRef.current = c;
      URL.revokeObjectURL(objUrl);
    };
    vid.onerror = () => URL.revokeObjectURL(objUrl);
    vid.load();
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = () => setDragOver(false);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProgress(null);

    try {
      let res;

      if (file.size > CHUNK_SIZE) {
        // --- Chunked upload with background processing ---
        // Processing starts on the server as soon as chunk 0 arrives.
        // We finish all uploads first, then connect to the SSE stream to
        // collect results (processing is already underway by then).
        const sessionId = crypto.randomUUID();
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        // Helper: read SSE events from a fetch Response, returns final result
        const readSSE = async (sseRes) => {
          const reader = sseRes.body.getReader();
          const dec = new TextDecoder();
          let buf = '';
          let sseResult = null;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += dec.decode(value, { stream: true });
            const parts = buf.split('\n\n');
            buf = parts.pop() || '';

            for (const block of parts) {
              if (!block.trim()) continue;
              let eventType = 'message';
              let dataStr = '';
              for (const line of block.split('\n')) {
                if (line.startsWith('event: ')) eventType = line.slice(7).trim();
                else if (line.startsWith('data: ')) dataStr = line.slice(6);
              }
              if (!dataStr) continue;
              const data = JSON.parse(dataStr);

              if (eventType === 'progress') {
                setProgress({ phase: 'processing', processing: data });
              } else if (eventType === 'complete') {
                sseResult = data;
              } else if (eventType === 'error') {
                throw new Error(data.detail || 'Processing failed on server');
              }
            }
          }
          return sseResult;
        };

        // 1. Upload all chunks (server starts processing on chunk 0)
        for (let i = 0; i < totalChunks; i++) {
          const start = i * CHUNK_SIZE;
          const chunk = file.slice(start, start + CHUNK_SIZE);

          const formData = new FormData();
          formData.append('session_id', sessionId);
          formData.append('chunk_index', i);
          formData.append('total_chunks', totalChunks);
          formData.append('ext', ext);
          formData.append('file', chunk, 'chunk.bin');
          if (plateColor && plateColor !== 'skip') {
            formData.append('plate_r', plateColor.r);
            formData.append('plate_g', plateColor.g);
            formData.append('plate_b', plateColor.b);
            console.log('[color-picker] sending plate color (chunked):', plateColor);
          }

          setProgress({
            phase: 'uploading',
            chunk: i + 1,
            totalChunks,
            percent: Math.round(((i + 1) / totalChunks) * 100),
          });

          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), CHUNK_TIMEOUT_MS);
          let uploadRes;
          try {
            uploadRes = await fetch(`${API_BASE}/api/upload-chunk`, {
              method: 'POST',
              body: formData,
              signal: controller.signal,
            });
          } catch (fetchErr) {
            if (fetchErr.name === 'AbortError') {
              throw new Error(`Chunk ${i + 1} timed out — network too slow or proxy blocking request`);
            }
            throw fetchErr;
          } finally {
            clearTimeout(timer);
          }

          if (!uploadRes.ok) {
            const body = await uploadRes.json().catch(() => null);
            throw new Error(body?.detail || `Upload error (${uploadRes.status})`);
          }

          // Check if server confirmed streaming is active
          const uploadData = await uploadRes.json().catch(() => ({}));
          if (uploadData.streaming) {
            // Processing already underway on server — note the phase shift
            setProgress(prev => prev?.phase === 'uploading' ? { ...prev, processingStarted: true } : prev);
          }
        }

        // 2. All uploads done — connect to SSE stream (processing already running)
        setProgress({ phase: 'processing', processing: null });

        const sseRes = await fetch(`${API_BASE}/api/stream-progress/${sessionId}`);

        if (sseRes.ok) {
          const sseResult = await readSSE(sseRes);
          setResult(sseResult);
        } else {
          // Fallback: stream-progress unavailable, use legacy endpoint
          res = await fetch(`${API_BASE}/api/analyze-assembled/${sessionId}`, {
            method: 'POST',
          });
          if (!res.ok) {
            const body = await res.json().catch(() => null);
            throw new Error(body?.detail || `Server error (${res.status})`);
          }
          const fallbackResult = await readSSE(res);
          setResult(fallbackResult);
        }
      } else {
        // --- Direct upload (small files) ---
        const formData = new FormData();
        formData.append('file', file);
        const streamUrl = plateColor && plateColor !== 'skip'
          ? `${API_BASE}/api/analyze-stream?plate_r=${plateColor.r}&plate_g=${plateColor.g}&plate_b=${plateColor.b}`
          : `${API_BASE}/api/analyze-stream`;
        console.log('[color-picker] sending plate color (stream):', plateColor, 'url:', streamUrl);
        res = await fetch(streamUrl, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail || `Server error (${res.status})`);
        }

        // Read SSE stream
        const reader = res.body.getReader();
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

            if (eventType === 'progress') setProgress({ phase: 'processing', ...data });
            else if (eventType === 'complete') setResult(data);
            else if (eventType === 'error') throw new Error(data.detail || 'Processing failed on server');
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
      setProgress(null);
    }
  };

  const reset = () => {
    setFile(null);
    setVideoPreview(null);
    setResult(null);
    setError(null);
    setProgress(null);
    setCalibFrame(null);
    setPlateColor(null);
    calibCanvasRef.current = null;
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Weightlifting Scene Analyzer</h1>
        <p>
          Upload a gym video to detect equipment and estimate lifter pose on every frame
        </p>
      </header>

      {!result && !loading && (
        <>
          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mkv"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <div className="upload-icon">&#127909;</div>
            <h3>Drop a video here or click to browse</h3>
            <p>Supports MP4, MOV, AVI, WebM, MKV</p>
          </div>

          {file && videoPreview && (
            <>
              <div className="preview-strip">
                <video src={videoPreview} className="preview-thumb" muted />
                <div className="preview-info">
                  <div className="name">{file.name}</div>
                  <div className="size">{formatBytes(file.size)}</div>
                </div>
                <button
                  className="analyze-btn"
                  onClick={analyze}
                  disabled={loading || (calibFrame && !plateColor)}
                  title={calibFrame && !plateColor ? 'Click a plate in the preview below first' : ''}
                >
                  Process Video
                </button>
                <button className="remove-btn" onClick={reset} title="Remove">
                  &times;
                </button>
              </div>

              {/* Color calibration — click on a plate to pick target hue */}
              {calibFrame && !plateColor && (
                <div className="calib-section">
                  <p className="calib-instructions">
                    Click directly on a plate in the frame below to set the detection color.
                    &nbsp;<button className="skip-calib-btn" onClick={() => setPlateColor('skip')}>Skip (use shape detection)</button>
                  </p>
                  <img
                    src={calibFrame}
                    alt="First frame — click a plate"
                    className="calib-frame"
                    onClick={(e) => {
                      const canvas = calibCanvasRef.current;
                      if (!canvas) return;
                      const rect = e.currentTarget.getBoundingClientRect();

                      // Account for object-fit:contain letterboxing
                      const imgAspect = canvas.width / canvas.height;
                      const boxAspect = rect.width / rect.height;
                      let renderedW, renderedH, padX, padY;
                      if (imgAspect > boxAspect) {
                        renderedW = rect.width;
                        renderedH = rect.width / imgAspect;
                        padX = 0;
                        padY = (rect.height - renderedH) / 2;
                      } else {
                        renderedH = rect.height;
                        renderedW = rect.height * imgAspect;
                        padX = (rect.width - renderedW) / 2;
                        padY = 0;
                      }

                      const x = Math.round(((e.clientX - rect.left - padX) / renderedW) * canvas.width);
                      const y = Math.round(((e.clientY - rect.top - padY) / renderedH) * canvas.height);

                      // Ignore clicks in the letterbox padding
                      if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return;

                      const [r, g, b] = canvas.getContext('2d').getImageData(x, y, 1, 1).data;
                      console.log('[color-picker] click pos:', { clientX: e.clientX, clientY: e.clientY },
                        'rect:', { left: rect.left, top: rect.top, w: rect.width, h: rect.height },
                        'rendered:', { renderedW, renderedH, padX, padY },
                        'canvas px:', { x, y }, 'rgb:', { r, g, b });
                      setPlateColor({ r, g, b });
                    }}
                  />
                </div>
              )}

              {plateColor && plateColor !== 'skip' && (
                <div className="calib-confirm">
                  <div
                    className="color-swatch"
                    style={{ background: `rgb(${plateColor.r},${plateColor.g},${plateColor.b})` }}
                  />
                  <span>Plate color set — detection will target this hue</span>
                  <button className="repick-btn" onClick={() => setPlateColor(null)}>Re-pick</button>
                </div>
              )}

              {plateColor === 'skip' && (
                <div className="calib-confirm calib-skipped">
                  <span>Using shape detection (no color calibration)</span>
                  <button className="repick-btn" onClick={() => setPlateColor(null)}>Calibrate instead</button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {loading && (
        <div className="processing">
          {/* Upload progress (shown during 'uploading' phase) */}
          {progress?.phase === 'uploading' && (
            <>
              <div className="spinner" />
              <p>Uploading chunk {progress.chunk} of {progress.totalChunks}...</p>
              <div className="progress-section">
                <div className="progress-bar-track">
                  <div className="progress-bar-fill" style={{ width: `${progress.percent}%` }} />
                </div>
                <div className="progress-text">{progress.percent}% uploaded</div>
              </div>
            </>
          )}
          {/* Processing preview (shown during 'processing' phase, chunked upload path) */}
          {progress?.phase === 'processing' && progress?.processing?.preview && (
            <>
              <div className="preview-container">
                <img
                  src={`data:image/jpeg;base64,${progress.processing.preview}`}
                  alt={`Processing frame ${progress.processing.frame_index}`}
                  className="live-preview"
                />
              </div>
              <div className="progress-section">
                {progress.processing.percent != null ? (
                  <>
                    <div className="progress-bar-track">
                      <div className="progress-bar-fill" style={{ width: `${progress.processing.percent}%` }} />
                    </div>
                    <div className="progress-text">
                      Frame {progress.processing.frame_index} of {progress.processing.total_frames} ({progress.processing.percent}%)
                    </div>
                  </>
                ) : (
                  <div className="progress-text">
                    Processing frame {progress.processing.frame_index}...
                  </div>
                )}
              </div>
            </>
          )}
          {/* Legacy processing view (direct upload / fallback) */}
          {(progress?.phase === 'processing' && progress.preview) && (
            <>
              <div className="preview-container">
                <img
                  src={`data:image/jpeg;base64,${progress.preview}`}
                  alt={`Processing frame ${progress.frame_index}`}
                  className="live-preview"
                />
              </div>
              <div className="progress-section">
                <div className="progress-bar-track">
                  <div className="progress-bar-fill" style={{ width: `${progress.percent}%` }} />
                </div>
                <div className="progress-text">
                  Frame {progress.frame_index} of {progress.total_frames} ({progress.percent}%)
                </div>
              </div>
            </>
          )}
          {!progress && (
            <>
              <div className="spinner" />
              <p>Uploading and initializing...</p>
            </>
          )}
          <p className="loading-subtext">
            Running detection and pose estimation on every frame.
          </p>
        </div>
      )}

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="results">
          <div className="reset-bar">
            <button className="reset-btn" onClick={reset}>
              Process Another Video
            </button>
          </div>
          <h2>Processing Complete</h2>

          <div className="video-stats">
            <div className="stat-card">
              <div className="stat-value">{result.total_frames}</div>
              <div className="stat-label">Frames Processed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.fps.toFixed(1)}</div>
              <div className="stat-label">FPS</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.duration_seconds.toFixed(1)}s</div>
              <div className="stat-label">Duration</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.width}x{result.height}</div>
              <div className="stat-label">Resolution</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.processing_time_seconds.toFixed(1)}s</div>
              <div className="stat-label">Processing Time</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.frames_with_pose}</div>
              <div className="stat-label">Frames with Pose</div>
            </div>
          </div>

          <div className="download-section">
            <a
              href={`${API_BASE}${result.download_url}`}
              download
              className="download-btn"
            >
              Download Annotated Video (MP4)
            </a>
          </div>

          <div className="detections-panel">
            <h3>Detection Summary (across all frames)</h3>
            {Object.keys(result.detection_summary).length === 0 ? (
              <div className="detection-item" style={{ color: 'var(--text-muted)' }}>
                No objects detected in any frame
              </div>
            ) : (
              <ul className="detection-list">
                {Object.entries(result.detection_summary)
                  .sort(([, a], [, b]) => b - a)
                  .map(([label, count]) => (
                    <li key={label} className="detection-item">
                      <span className="det-label">{label}</span>
                      <span className="det-conf conf-high">
                        {count} frames
                      </span>
                    </li>
                  ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
