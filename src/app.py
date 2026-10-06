"""
Standalone Local Web Server for Kirana Product Recognition Demo.
Built exclusively using Python's standard-library http.server.
Requires zero external web frameworks (no Flask, FastAPI, Streamlit, etc.).
"""

import sys
import json
import io
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Set project path
PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import Predictor

# Global Predictor instance loaded once on startup
print("Initializing Kirana Product Recognition Engine...")
PREDICTOR = Predictor()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kirana Product Recognition | CNN Demo</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --primary-light: #dbeafe;
      --success: #16a34a;
      --success-light: #dcfce7;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 12px;
      --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 24px 16px;
    }

    .container {
      max-width: 860px;
      margin: 0 auto;
      width: 100%;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 28px;
    }

    .badge {
      display: inline-block;
      background: var(--primary-light);
      color: var(--primary);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 4px 12px;
      border-radius: 9999px;
      margin-bottom: 8px;
    }

    h1 {
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.02em;
      margin-bottom: 6px;
    }

    .subtitle {
      font-size: 1.05rem;
      color: var(--text-muted);
    }

    /* Grid Layout */
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }

    @media (max-width: 720px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
    }

    .card-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Upload Area */
    .dropzone {
      border: 2px dashed #cbd5e1;
      border-radius: 10px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: #fafafa;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: var(--primary-light);
    }

    .dropzone-icon {
      font-size: 2rem;
      margin-bottom: 8px;
      color: var(--primary);
    }

    .dropzone-text {
      font-size: 0.95rem;
      color: var(--text);
      font-weight: 600;
      margin-bottom: 4px;
    }

    .dropzone-hint {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    input[type="file"] {
      display: none;
    }

    /* Preview */
    .preview-container {
      display: none;
      margin-top: 14px;
      text-align: center;
    }

    .preview-img {
      max-width: 100%;
      max-height: 200px;
      border-radius: 8px;
      border: 1px solid var(--border);
      object-fit: contain;
      box-shadow: var(--shadow);
    }

    .file-details {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 6px;
      word-break: break-all;
    }

    /* Action Buttons */
    .btn-group {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }

    .btn {
      flex: 1;
      padding: 11px 16px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
      display: inline-flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
    }

    .btn-primary {
      background: var(--primary);
      color: white;
    }

    .btn-primary:hover:not(:disabled) {
      background: var(--primary-hover);
      box-shadow: var(--shadow);
    }

    .btn-primary:disabled {
      background: #94a3b8;
      cursor: not-allowed;
    }

    .btn-secondary {
      background: #f1f5f9;
      color: #475569;
      flex: 0 0 auto;
    }

    .btn-secondary:hover:not(:disabled) {
      background: #e2e8f0;
    }

    /* Results */
    .placeholder-state {
      text-align: center;
      color: var(--text-muted);
      padding: 40px 10px;
    }

    .placeholder-icon {
      font-size: 2.2rem;
      margin-bottom: 10px;
      opacity: 0.5;
    }

    .result-content {
      display: none;
    }

    .prediction-highlight {
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 16px;
      text-align: center;
      margin-bottom: 20px;
    }

    .prediction-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--primary);
      font-weight: 700;
      margin-bottom: 4px;
    }

    .prediction-value {
      font-size: 2rem;
      font-weight: 800;
      color: #1e3a8a;
    }

    .prediction-conf {
      font-size: 0.95rem;
      color: #3b82f6;
      font-weight: 600;
      margin-top: 2px;
    }

    /* Probability Bars */
    .prob-item {
      margin-bottom: 12px;
    }

    .prob-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.88rem;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .prob-bar-bg {
      height: 10px;
      background: #e2e8f0;
      border-radius: 9999px;
      overflow: hidden;
    }

    .prob-bar-fill {
      height: 100%;
      background: var(--primary);
      border-radius: 9999px;
      width: 0%;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .prob-bar-fill.highest {
      background: var(--success);
    }

    /* Error Banner */
    .error-banner {
      display: none;
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #dc2626;
      padding: 12px 14px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 500;
      margin-bottom: 16px;
    }

    /* Model Info Section */
    .info-card {
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 22px;
      box-shadow: var(--shadow);
    }

    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px;
      margin-top: 10px;
    }

    .info-item {
      background: #f8fafc;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid #f1f5f9;
    }

    .info-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.04em;
    }

    .info-value {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 2px;
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge">Live Demonstration</span>
      <h1>Kirana Product Recognition</h1>
      <p class="subtitle">CNN-Based Product Classification</p>
    </header>

    <div id="errorBanner" class="error-banner"></div>

    <div class="grid">
      <!-- Upload Panel -->
      <div class="card">
        <h2 class="card-title">1. Upload Product Image</h2>
        <div id="dropzone" class="dropzone">
          <div class="dropzone-icon">📷</div>
          <div class="dropzone-text">Click or drag & drop image here</div>
          <div class="dropzone-hint">Supports JPEG, PNG, WEBP (Juice, Milk, or Yoghurt)</div>
          <input type="file" id="fileInput" accept="image/*">
        </div>

        <div id="previewContainer" class="preview-container">
          <img id="previewImg" class="preview-img" alt="Product Preview">
          <div id="fileDetails" class="file-details"></div>
        </div>

        <div class="btn-group">
          <button id="predictBtn" class="btn btn-primary" disabled>
            <span id="btnIcon">⚡</span>
            <span id="btnText">Predict Product</span>
          </button>
          <button id="clearBtn" class="btn btn-secondary" title="Clear upload" disabled>✕ Clear</button>
        </div>
      </div>

      <!-- Results Panel -->
      <div class="card">
        <h2 class="card-title">2. Classification Result</h2>
        
        <div id="placeholderState" class="placeholder-state">
          <div class="placeholder-icon">🏷️</div>
          <p>Upload a product image and click <strong>Predict Product</strong> to see real-time CNN output.</p>
        </div>

        <div id="resultContent" class="result-content">
          <div class="prediction-highlight">
            <div class="prediction-label">Predicted Class</div>
            <div id="predClass" class="prediction-value">—</div>
            <div id="predConf" class="prediction-conf">—% Confidence</div>
          </div>

          <h3 style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Class Probabilities</h3>
          
          <div id="probBars"></div>
        </div>
      </div>
    </div>

    <!-- Model Specs Panel -->
    <div class="info-card">
      <h3 style="font-size: 1rem; font-weight: 700; margin-bottom: 4px;">Model Information</h3>
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">Architecture</div>
          <div class="info-value">Baseline CNN</div>
        </div>
        <div class="info-item">
          <div class="info-label">Input Size</div>
          <div class="info-value">128 × 128 RGB</div>
        </div>
        <div class="info-item">
          <div class="info-label">Classes</div>
          <div class="info-value">3 Categories</div>
        </div>
        <div class="info-item">
          <div class="info-label">Validation Acc</div>
          <div class="info-value" style="color: var(--primary);">96.30%</div>
        </div>
        <div class="info-item">
          <div class="info-label">Test Acc</div>
          <div class="info-value" style="color: var(--success);">94.44%</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImg = document.getElementById("previewImg");
    const fileDetails = document.getElementById("fileDetails");
    const predictBtn = document.getElementById("predictBtn");
    const clearBtn = document.getElementById("clearBtn");
    const btnText = document.getElementById("btnText");
    const btnIcon = document.getElementById("btnIcon");
    const errorBanner = document.getElementById("errorBanner");
    const placeholderState = document.getElementById("placeholderState");
    const resultContent = document.getElementById("resultContent");
    const predClass = document.getElementById("predClass");
    const predConf = document.getElementById("predConf");
    const probBars = document.getElementById("probBars");

    let currentFile = null;

    function showError(msg) {
      errorBanner.textContent = msg;
      errorBanner.style.display = "block";
    }

    function clearError() {
      errorBanner.textContent = "";
      errorBanner.style.display = "none";
    }

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });

    function handleFile(file) {
      clearError();
      if (!file.type.startsWith("image/")) {
        showError("Unsupported file type. Please upload an image file (JPEG, PNG, WEBP).");
        return;
      }

      currentFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.style.display = "block";
        dropzone.style.display = "none";
        fileDetails.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        predictBtn.disabled = false;
        clearBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }

    function resetUI() {
      currentFile = null;
      fileInput.value = "";
      previewContainer.style.display = "none";
      dropzone.style.display = "block";
      predictBtn.disabled = true;
      clearBtn.disabled = true;
      placeholderState.style.display = "block";
      resultContent.style.display = "none";
      clearError();
    }

    clearBtn.addEventListener("click", resetUI);

    predictBtn.addEventListener("click", async () => {
      if (!currentFile) {
        showError("Please select an image first.");
        return;
      }

      clearError();
      predictBtn.disabled = true;
      clearBtn.disabled = true;
      btnText.textContent = "Analyzing...";
      btnIcon.innerHTML = '<span class="spinner"></span>';

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "Content-Type": currentFile.type || "application/octet-stream"
          },
          body: currentFile
        });

        const data = await response.json();

        if (!data.success) {
          showError(data.error || "Prediction failed.");
          return;
        }

        renderResults(data);
      } catch (err) {
        showError("Unable to reach prediction server. Please verify the server is running.");
      } finally {
        predictBtn.disabled = false;
        clearBtn.disabled = false;
        btnText.textContent = "Predict Product";
        btnIcon.textContent = "⚡";
      }
    });

    function renderResults(data) {
      placeholderState.style.display = "none";
      resultContent.style.display = "block";

      predClass.textContent = data.predicted_class;
      predConf.textContent = `${data.confidence.toFixed(2)}% Confidence`;

      probBars.innerHTML = "";
      const probs = data.probabilities;
      const maxVal = Math.max(...Object.values(probs));

      for (const [cls, val] of Object.entries(probs)) {
        const isHighest = (val === maxVal);
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
          <div class="prob-header">
            <span>${cls}</span>
            <span>${val.toFixed(2)}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill ${isHighest ? 'highest' : ''}" style="width: 0%"></div>
          </div>
        `;
        probBars.appendChild(item);
        // Animate width
        setTimeout(() => {
          item.querySelector(".prob-bar-fill").style.width = `${val}%`;
        }, 50);
      }
    }
  </script>
</body>
</html>
"""

class KiranaHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} -> {args[1]}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML_CONTENT.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/info":
            info = {
                "name": "Kirana Product Recognition",
                "classes": PREDICTOR.class_names,
                "input_shape": [128, 128, 3],
                "validation_accuracy": 96.30,
                "test_accuracy": 94.44
            }
            body = json.dumps(info).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._send_json({"success": False, "error": "No image data sent in request body."}, 400)
                    return

                raw_bytes = self.rfile.read(content_length)

                # Check if multipart/form-data
                content_type = self.headers.get("Content-Type", "")
                image_bytes = raw_bytes
                if "multipart/form-data" in content_type:
                    # Extract file payload from multipart
                    boundary = content_type.split("boundary=")[-1].strip()
                    parts = raw_bytes.split(f"--{boundary}".encode())
                    for p in parts:
                        if b"Content-Type: image/" in p or b"filename=" in p:
                            header_end = p.find(b"\r\n\r\n")
                            if header_end != -1:
                                image_bytes = p[header_end + 4:].rstrip(b"\r\n")
                                break

                # Run prediction
                result = PREDICTOR.predict_bytes(image_bytes)
                status_code = 200 if result.get("success") else 400
                self._send_json(result, status_code)
            except Exception as e:
                self._send_json({"success": False, "error": f"Server processing error: {str(e)}"}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def run_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, KiranaHTTPHandler)
    print("=" * 65)
    print(f"Kirana Product Recognition Web UI active!")
    print(f"URL: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
