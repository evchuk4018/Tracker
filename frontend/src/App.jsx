import { useState, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = '';

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function confClass(conf) {
  if (conf >= 0.75) return 'conf-high';
  if (conf >= 0.5) return 'conf-mid';
  return 'conf-low';
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowed.includes(f.type)) {
      setError('Please upload a JPEG, PNG, or WebP image.');
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      setError('File is too large (max 20 MB).');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
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

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Server error (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Weightlifting Scene Analyzer</h1>
        <p>
          Upload a gym photo to detect equipment and estimate lifter pose
        </p>
      </header>

      {!result && (
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
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <div className="upload-icon">&#128247;</div>
            <h3>Drop an image here or click to browse</h3>
            <p>Supports JPEG, PNG, WebP up to 20 MB</p>
          </div>

          {file && preview && (
            <div className="preview-strip">
              <img src={preview} alt="preview" className="preview-thumb" />
              <div className="preview-info">
                <div className="name">{file.name}</div>
                <div className="size">{formatBytes(file.size)}</div>
              </div>
              <button className="analyze-btn" onClick={analyze} disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
              <button className="remove-btn" onClick={reset} title="Remove">
                &times;
              </button>
            </div>
          )}
        </>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Running object detection and pose estimation...</p>
        </div>
      )}

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="results">
          <div className="reset-bar">
            <button className="reset-btn" onClick={reset}>
              New Analysis
            </button>
          </div>
          <h2>Analysis Results</h2>
          <div className="results-grid">
            <div>
              <div className="result-image-container">
                <img
                  src={`${API_BASE}${result.annotated_image_url}`}
                  alt="Annotated result"
                />
              </div>
            </div>
            <div>
              <div className="detections-panel">
                <h3>Detections ({result.detections.length})</h3>
                {result.detections.length === 0 ? (
                  <div className="detection-item" style={{ color: 'var(--text-muted)' }}>
                    No objects detected
                  </div>
                ) : (
                  <ul className="detection-list">
                    {result.detections.map((det, i) => (
                      <li key={i} className="detection-item">
                        <span className="det-label">{det.label}</span>
                        <span className={`det-conf ${confClass(det.confidence)}`}>
                          {(det.confidence * 100).toFixed(0)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div className="pose-panel">
                {result.pose ? (
                  <>
                    <strong>Pose Detected</strong>
                    <br />
                    {result.pose.keypoints.filter((k) => k.visibility > 0.5).length} of{' '}
                    {result.pose.keypoints.length} keypoints visible
                  </>
                ) : (
                  'No human pose detected in this image.'
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
