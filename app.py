"""
Flask backend for real-time web object detection.
Accepts base64 frames from browser, returns detection JSON, serves detection history.
"""
import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from detector import ObjectDetectionService
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Real-Time Object Detection</title>
<style>
  body { font-family: Arial, sans-serif; margin: 0; background: #111; color: #eee; }
  #container { display: flex; gap: 16px; padding: 16px; }
  #videoSection { flex: 1; }
  #sidebar { width: 280px; }
  video, canvas { border: 2px solid #444; border-radius: 6px; max-width: 100%; }
  canvas { position: absolute; top: 0; left: 0; }
  .det-box { background: #1e2a3a; border-radius: 4px; padding: 8px; margin: 4px 0; font-size: 13px; }
  .label { font-weight: bold; color: #5af; }
  button { background: #2563eb; color: white; border: none; padding: 8px 18px; border-radius: 4px; cursor: pointer; margin: 4px; }
  button:hover { background: #1d4ed8; }
  #status { font-size: 12px; color: #aaa; margin-top: 8px; }
</style>
</head>
<body>
<h2 style="padding:16px 16px 0;margin:0">Real-Time Object Detection</h2>
<div id="container">
  <div id="videoSection">
    <div style="position:relative;display:inline-block">
      <video id="video" width="640" height="480" autoplay muted></video>
      <canvas id="overlay" width="640" height="480"></canvas>
    </div>
    <br>
    <button onclick="startCamera()">Start Camera</button>
    <button onclick="stopCamera()">Stop</button>
    <div id="status">Camera not started</div>
  </div>
  <div id="sidebar">
    <h3>Detections</h3>
    <div id="detections">No detections yet</div>
    <h3>Stats</h3>
    <div id="stats">--</div>
  </div>
</div>
<script>
let stream = null;
let intervalId = null;
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const COLORS = ['#f87171','#34d399','#60a5fa','#fbbf24','#a78bfa','#fb923c'];

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    document.getElementById('status').textContent = 'Camera active - detecting every 500ms';
    intervalId = setInterval(captureAndDetect, 500);
  } catch(e) {
    document.getElementById('status').textContent = 'Camera error: ' + e.message;
  }
}

function stopCamera() {
  if(stream) stream.getTracks().forEach(t => t.stop());
  if(intervalId) clearInterval(intervalId);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  document.getElementById('status').textContent = 'Camera stopped';
}

async function captureAndDetect() {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = video.videoWidth || 640;
  tmpCanvas.height = video.videoHeight || 480;
  tmpCanvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = tmpCanvas.toDataURL('image/jpeg', 0.7);
  try {
    const resp = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: b64 })
    });
    const data = await resp.json();
    renderBoxes(data.detections || []);
    renderSidebar(data);
  } catch(e) { console.error(e); }
}

function renderBoxes(detections) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  detections.forEach((d, i) => {
    const color = COLORS[i % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    ctx.fillStyle = color;
    ctx.font = '13px Arial';
    ctx.fillText(d.label + ' ' + (d.confidence * 100).toFixed(0) + '%', d.x1 + 2, d.y1 - 4);
  });
}

function renderSidebar(data) {
  const detDiv = document.getElementById('detections');
  if(!data.detections || data.detections.length === 0) {
    detDiv.innerHTML = '<p style="color:#888">No objects detected</p>';
  } else {
    detDiv.innerHTML = data.detections.map((d,i) =>
      '<div class="det-box"><span class="label">' + d.label + '</span> ' +
      (d.confidence*100).toFixed(1) + '% &nbsp;|&nbsp; ' +
      d.width.toFixed(0) + 'x' + d.height.toFixed(0) + 'px</div>'
    ).join('');
  }
  document.getElementById('stats').innerHTML =
    '<b>Inference:</b> ' + (data.inference_ms || 0).toFixed(1) + 'ms<br>' +
    '<b>Count:</b> ' + (data.count || 0) + ' objects<br>' +
    '<b>Frame:</b> ' + (data.frame_id || '--');
}
</script>
</body>
</html>
"""

_service = None


def get_service() -> "ObjectDetectionService":
    global _service
    if _service is None and DETECTOR_AVAILABLE:
        _service = ObjectDetectionService(
            model_path="yolov8n.pt",
            confidence_threshold=0.45,
            db_path="detections.db",
        )
    return _service


def create_app() -> "Flask":
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask is required: pip install flask")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(DEMO_HTML)

    @app.route("/detect", methods=["POST"])
    def detect():
        data = request.get_json(silent=True) or {}
        b64_frame = data.get("frame", "")
        service = get_service()
        if service is None:
            return jsonify({"error": "detector not available", "count": 0, "detections": []}), 503
        try:
            result = service.process_b64_frame(b64_frame)
            return jsonify(result.to_dict())
        except Exception as exc:
            logger.error("Detection error: %s", exc)
            return jsonify({"error": str(exc), "count": 0, "detections": []}), 500

    @app.route("/history")
    def history():
        service = get_service()
        if service is None:
            return jsonify([])
        hours = float(request.args.get("hours", 1.0))
        counts = service.store.recent_counts(hours)
        return jsonify(counts)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "timestamp": time.time()})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not FLASK_AVAILABLE:
        print("Flask not installed. Install: pip install flask")
    else:
        app = create_app()
        print("Starting object detection server on http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
