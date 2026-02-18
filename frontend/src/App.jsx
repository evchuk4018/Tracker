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
        res = await fetch(`${API_BASE}/api/analyze-stream`, {
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
            <div className="preview-strip">
              <video src={videoPreview} className="preview-thumb" muted />
              <div className="preview-info">
                <div className="name">{file.name}</div>
                <div className="size">{formatBytes(file.size)}</div>
              </div>
              <button className="analyze-btn" onClick={analyze} disabled={loading}>
                Process Video
              </button>
              <button className="remove-btn" onClick={reset} title="Remove">
                &times;
              </button>
            </div>
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
