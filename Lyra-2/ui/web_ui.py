# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small browser UI for running Lyra 2 inference jobs.

The model stack is intentionally kept in the existing command-line inference
modules. This server only handles uploads, subprocess orchestration, job status,
and serving generated artifacts.
"""

from __future__ import annotations

import cgi
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


ROOT = Path(os.environ.get("LYRA2_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUT_ROOT = Path(os.environ.get("LYRA2_OUTPUT_DIR", ROOT / "outputs" / "ui_jobs")).resolve()
HOST = os.environ.get("LYRA2_HOST", "0.0.0.0")
PORT = int(os.environ.get("LYRA2_PORT", "8080"))
MAX_UPLOAD_BYTES = int(os.environ.get("LYRA2_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
LOG_LIMIT = int(os.environ.get("LYRA2_UI_LOG_LIMIT", "2000"))
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Job:
    id: str
    mode: str
    job_dir: Path
    output_dir: Path
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    status: str = "queued"
    message: str = "Queued"
    command: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    returncode: int | None = None
    video_path: Path | None = None
    gs_video_path: Path | None = None
    ply_path: Path | None = None
    process: subprocess.Popen[str] | None = None

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        if len(self.logs) > LOG_LIMIT:
            del self.logs[: len(self.logs) - LOG_LIMIT]

    def public_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "returncode": self.returncode,
            "command": self.command,
            "video_url": _file_url(self, self.video_path),
            "gs_video_url": _file_url(self, self.gs_video_path),
            "ply_url": _file_url(self, self.ply_path),
            "logs": "\n".join(self.logs[-LOG_LIMIT:]),
        }


STATE_LOCK = threading.Lock()
CURRENT_JOB: Job | None = None
LAST_JOB: Job | None = None


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    *,
    status: HTTPStatus = HTTPStatus.OK,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _field_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    item = form[name] if name in form else None
    if item is None or isinstance(item, list) or item.filename:
        return default
    value = item.value
    return value if isinstance(value, str) else default


def _bool_field(form: cgi.FieldStorage, name: str) -> bool:
    return _field_value(form, name).lower() in {"1", "true", "yes", "on"}


def _int_field(form: cgi.FieldStorage, name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(_field_value(form, name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _float_field(form: cgi.FieldStorage, name: str, default: float) -> float:
    try:
        return float(_field_value(form, name, str(default)))
    except ValueError:
        return default


def _safe_filename(name: str, fallback: str) -> str:
    candidate = Path(name or fallback).name
    candidate = SAFE_STEM_RE.sub("_", candidate).strip("._")
    return candidate or fallback


def _save_upload(form: cgi.FieldStorage, field_name: str, directory: Path, fallback: str) -> Path | None:
    item = form[field_name] if field_name in form else None
    if item is None or isinstance(item, list) or not item.filename:
        return None
    path = directory / _safe_filename(item.filename, fallback)
    with path.open("wb") as handle:
        shutil.copyfileobj(item.file, handle)
    return path


def _file_url(job: Job, path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        rel = path.resolve().relative_to(job.job_dir.resolve())
    except ValueError:
        return None
    return f"/files/{job.id}/{quote(str(rel))}"


def _resolve_served_file(job_id: str, rel_url_path: str) -> Path | None:
    job = None
    with STATE_LOCK:
        for candidate in (CURRENT_JOB, LAST_JOB):
            if candidate and candidate.id == job_id:
                job = candidate
                break
    if job is None:
        return None
    rel_path = Path(unquote(rel_url_path))
    try:
        candidate = (job.job_dir / rel_path).resolve()
        candidate.relative_to(job.job_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".ply":
        return "application/octet-stream"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return "application/octet-stream"


def _common_args(form: cgi.FieldStorage) -> list[str]:
    args = [
        "--experiment",
        _field_value(form, "experiment", "lyra2"),
        "--checkpoint_dir",
        _field_value(form, "checkpoint_dir", "checkpoints/model"),
        "--resolution",
        _field_value(form, "resolution", "480,832"),
        "--fps",
        str(_int_field(form, "fps", 16, minimum=1)),
        "--seed",
        str(_int_field(form, "seed", 1)),
        "--guidance",
        str(_float_field(form, "guidance", 5.0)),
        "--shift",
        str(_float_field(form, "shift", 5.0)),
    ]
    if _bool_field(form, "use_dmd"):
        args.append("--use_dmd")
    if _bool_field(form, "offload"):
        args.append("--offload")
    if _bool_field(form, "offload_when_prompt"):
        args.append("--offload_when_prompt")
    if not _bool_field(form, "use_moge_scale"):
        args.append("--no-use_moge_scale")
    return args


def _build_job(form: cgi.FieldStorage) -> Job:
    mode = _field_value(form, "mode", "zoom")
    if mode not in {"zoom", "custom"}:
        raise ValueError("mode must be 'zoom' or 'custom'")

    prompt = _field_value(form, "prompt").strip()
    if not prompt:
        raise ValueError("prompt is required")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    upload_dir = job_dir / "uploads"
    output_dir = job_dir / "output"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _save_upload(form, "image", upload_dir, "input.png")
    if image_path is None:
        raise ValueError("image upload is required")
    image_stem = image_path.stem

    job = Job(id=job_id, mode=mode, job_dir=job_dir, output_dir=output_dir)
    common_args = _common_args(form)

    if mode == "zoom":
        cmd = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.lyra2_zoomgs_inference",
            "--input_image_path",
            str(image_path),
            "--num_samples",
            "1",
            "--prompt",
            prompt,
            "--output_path",
            str(output_dir),
            "--num_frames_zoom_in",
            str(_int_field(form, "num_frames_zoom_in", 81, minimum=1)),
            "--num_frames_zoom_out",
            str(_int_field(form, "num_frames_zoom_out", 241, minimum=1)),
            "--zoom_in_strength",
            str(_float_field(form, "zoom_in_strength", 0.5)),
            "--zoom_out_strength",
            str(_float_field(form, "zoom_out_strength", 1.5)),
            *common_args,
        ]
        if _bool_field(form, "ground_plane_align"):
            cmd.append("--ground_plane_align")
        job.video_path = output_dir / "videos" / f"{image_stem}.mp4"
    else:
        trajectory_path = _save_upload(form, "trajectory", upload_dir, "trajectory.npz")
        if trajectory_path is None:
            raise ValueError("trajectory .npz upload is required for custom trajectory mode")
        captions_path = _save_upload(form, "captions", upload_dir, "captions.json")
        cmd = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.lyra2_custom_traj_inference",
            "--input_image_path",
            str(image_path),
            "--trajectory_path",
            str(trajectory_path),
            "--num_samples",
            "1",
            "--prompt",
            prompt,
            "--output_path",
            str(output_dir),
            "--num_frames",
            str(_int_field(form, "num_frames", 161, minimum=1)),
            "--pose_scale",
            str(_float_field(form, "pose_scale", 1.1)),
            *common_args,
        ]
        if captions_path is not None:
            cmd.extend(["--captions_path", str(captions_path)])
        job.video_path = output_dir / f"{image_stem}.mp4"

    job.command = cmd
    if _bool_field(form, "reconstruct_gs"):
        job.append_log("[ui] Gaussian splatting reconstruction will run after video generation.")
    return job


def _run_subprocess(job: Job, cmd: list[str]) -> int:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    job.append_log("[ui] " + " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job.process = process
    assert process.stdout is not None
    for line in process.stdout:
        job.append_log(line)
    return process.wait()


def _run_job(job: Job, reconstruct_gs: bool) -> None:
    global CURRENT_JOB, LAST_JOB
    with STATE_LOCK:
        job.status = "running"
        job.message = "Running Lyra 2 inference"
        job.started_at = time.time()

    try:
        returncode = _run_subprocess(job, job.command)
        job.returncode = returncode
        if returncode != 0:
            raise RuntimeError(f"Lyra 2 inference exited with code {returncode}")
        if job.video_path is None or not job.video_path.exists():
            raise RuntimeError("Lyra 2 finished, but the expected output video was not found")

        if reconstruct_gs:
            job.message = "Running Gaussian splatting reconstruction"
            recon_cmd = [
                sys.executable,
                "-m",
                "lyra_2._src.inference.vipe_da3_gs_recon",
                "--input_video_path",
                str(job.video_path),
            ]
            recon_code = _run_subprocess(job, recon_cmd)
            if recon_code != 0:
                raise RuntimeError(f"GS reconstruction exited with code {recon_code}")
            recon_dir = job.video_path.with_name(f"{job.video_path.stem}_gs_ours")
            job.gs_video_path = recon_dir / "gs_trajectory.mp4"
            job.ply_path = recon_dir / "reconstructed_scene.ply"

        job.status = "succeeded"
        job.message = "Done"
    except Exception as exc:  # noqa: BLE001 - surface failures through UI.
        job.status = "failed"
        job.message = str(exc)
        job.append_log(f"[ui] ERROR: {exc}")
    finally:
        job.ended_at = time.time()
        job.process = None
        with STATE_LOCK:
            LAST_JOB = job
            if CURRENT_JOB and CURRENT_JOB.id == job.id:
                CURRENT_JOB = None


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lyra 2 Playground</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #0b1020; color: #e5e7eb; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }
    h1 { margin: 0 0 8px; font-size: 34px; }
    p { color: #a7b0c0; line-height: 1.55; }
    form, section { background: #111827; border: 1px solid #263247; border-radius: 18px; padding: 22px; margin-top: 20px; box-shadow: 0 18px 60px rgba(0,0,0,.28); }
    label { display: block; font-size: 13px; color: #cbd5e1; margin: 12px 0 6px; }
    input, textarea, select { width: 100%; box-sizing: border-box; border: 1px solid #334155; border-radius: 10px; background: #0f172a; color: #f8fafc; padding: 10px 12px; }
    textarea { min-height: 94px; resize: vertical; }
    input[type="checkbox"] { width: auto; margin-right: 8px; }
    button { border: 0; border-radius: 999px; background: linear-gradient(135deg, #7c3aed, #06b6d4); color: white; padding: 12px 22px; font-weight: 700; cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .grid.two { grid-template-columns: repeat(2, 1fr); }
    .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-top: 14px; }
    .hint { font-size: 12px; color: #94a3b8; }
    .hidden { display: none; }
    .status { font-weight: 700; }
    pre { max-height: 440px; overflow: auto; background: #020617; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; white-space: pre-wrap; }
    video { width: 100%; max-height: 520px; border-radius: 14px; background: #020617; }
    a { color: #67e8f9; }
    @media (max-width: 860px) { .grid, .grid.two { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Lyra 2 Playground</h1>
  <p>Upload a starting image, describe the scene, and launch the existing Lyra 2 inference pipeline. Jobs run one at a time on this machine's GPU. For quick experiments, enable DMD and use lower frame counts.</p>

  <form id="jobForm">
    <div class="grid two">
      <div>
        <label>Mode</label>
        <select name="mode" id="mode">
          <option value="zoom">Preset zoom-in / zoom-out</option>
          <option value="custom">Custom trajectory</option>
        </select>
      </div>
      <div>
        <label>Starting image</label>
        <input name="image" type="file" accept="image/png,image/jpeg,image/webp" required>
      </div>
    </div>

    <label>Prompt / caption</label>
    <textarea name="prompt" required placeholder="A detailed description of the starting view and desired world style"></textarea>

    <div id="zoomFields">
      <div class="grid">
        <div><label>Zoom-in frames</label><input name="num_frames_zoom_in" type="number" min="1" value="81"></div>
        <div><label>Zoom-out frames</label><input name="num_frames_zoom_out" type="number" min="1" value="241"></div>
        <div><label>Resolution H,W</label><input name="resolution" value="480,832"></div>
      </div>
      <div class="grid two">
        <div><label>Zoom-in strength</label><input name="zoom_in_strength" type="number" step="0.05" value="0.5"></div>
        <div><label>Zoom-out strength</label><input name="zoom_out_strength" type="number" step="0.05" value="1.5"></div>
      </div>
    </div>

    <div id="customFields" class="hidden">
      <div class="grid">
        <div><label>Trajectory .npz</label><input name="trajectory" type="file" accept=".npz"></div>
        <div><label>Captions JSON (optional)</label><input name="captions" type="file" accept=".json"></div>
        <div><label>Frames</label><input name="num_frames" type="number" min="1" value="161"></div>
      </div>
      <div class="grid two">
        <div><label>Pose scale</label><input name="pose_scale" type="number" step="0.05" value="1.1"></div>
        <div><label>Resolution H,W</label><input name="resolution_custom" value="480,832" oninput="document.querySelector('[name=resolution]').value=this.value"></div>
      </div>
    </div>

    <div class="grid">
      <div><label>Experiment</label><input name="experiment" value="lyra2"></div>
      <div><label>Checkpoint dir</label><input name="checkpoint_dir" value="checkpoints/model"></div>
      <div><label>Seed</label><input name="seed" type="number" value="1"></div>
    </div>
    <div class="grid">
      <div><label>FPS</label><input name="fps" type="number" min="1" value="16"></div>
      <div><label>Guidance</label><input name="guidance" type="number" step="0.1" value="5.0"></div>
      <div><label>Shift</label><input name="shift" type="number" step="0.1" value="5.0"></div>
    </div>

    <div class="row">
      <label><input name="use_dmd" type="checkbox" checked>Use DMD fast inference</label>
      <label><input name="use_moge_scale" type="checkbox" checked>Use MoGe scale alignment</label>
      <label><input name="offload" type="checkbox">Enable model offload</label>
      <label><input name="offload_when_prompt" type="checkbox">Offload prompt encoder</label>
      <label><input name="ground_plane_align" type="checkbox">Ground-plane align zoom</label>
      <label><input name="reconstruct_gs" type="checkbox">Run GS reconstruction after video</label>
    </div>
    <p class="hint">Frame counts work best as 1 + 80k, such as 81, 161, 241, or 481.</p>
    <button id="submitBtn" type="submit">Start Lyra 2 job</button>
  </form>

  <section>
    <div class="row"><span>Status:</span><span class="status" id="status">idle</span><button id="cancelBtn" type="button">Cancel running job</button></div>
    <p id="message"></p>
    <div id="result"></div>
    <pre id="logs"></pre>
  </section>
</main>
<script>
const form = document.getElementById('jobForm');
const mode = document.getElementById('mode');
const zoomFields = document.getElementById('zoomFields');
const customFields = document.getElementById('customFields');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const messageEl = document.getElementById('message');
const logsEl = document.getElementById('logs');
const resultEl = document.getElementById('result');

mode.addEventListener('change', () => {
  const custom = mode.value === 'custom';
  zoomFields.classList.toggle('hidden', custom);
  customFields.classList.toggle('hidden', !custom);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  const data = new FormData(form);
  if (data.get('resolution_custom') && mode.value === 'custom') {
    data.set('resolution', data.get('resolution_custom'));
  }
  const response = await fetch('/api/jobs', { method: 'POST', body: data });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || 'Failed to start job');
    submitBtn.disabled = false;
    return;
  }
  render(payload);
});

document.getElementById('cancelBtn').addEventListener('click', async () => {
  await fetch('/api/cancel', { method: 'POST' });
  await poll();
});

function render(payload) {
  if (!payload) return;
  statusEl.textContent = payload.status || 'idle';
  messageEl.textContent = payload.message || '';
  logsEl.textContent = payload.logs || '';
  logsEl.scrollTop = logsEl.scrollHeight;
  submitBtn.disabled = ['queued', 'running'].includes(payload.status);
  const parts = [];
  if (payload.video_url) {
    parts.push(`<h3>Generated video</h3><video src="${payload.video_url}" controls loop></video><p><a href="${payload.video_url}" download>Download MP4</a></p>`);
  }
  if (payload.gs_video_url) {
    parts.push(`<h3>GS flythrough</h3><video src="${payload.gs_video_url}" controls loop></video><p><a href="${payload.gs_video_url}" download>Download GS MP4</a></p>`);
  }
  if (payload.ply_url) {
    parts.push(`<p><a href="${payload.ply_url}" download>Download reconstructed_scene.ply</a></p>`);
  }
  resultEl.innerHTML = parts.join('');
}

async function poll() {
  const response = await fetch('/api/status');
  if (response.ok) render(await response.json());
}
setInterval(poll, 2500);
poll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Lyra2UI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            _text_response(self, HTML_PAGE, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/healthz":
            _json_response(self, {"ok": True})
            return
        if parsed.path == "/api/status":
            with STATE_LOCK:
                job = CURRENT_JOB or LAST_JOB
                payload = job.public_payload() if job else {"status": "idle", "message": "No jobs yet", "logs": ""}
            _json_response(self, payload)
            return
        if parsed.path.startswith("/files/"):
            parts = parsed.path.split("/", 3)
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = _resolve_served_file(parts[2], parts[3])
            if path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _mime_type(path))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/jobs":
            self._handle_create_job()
            return
        if self.path == "/api/cancel":
            self._handle_cancel()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_create_job(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_UPLOAD_BYTES:
            _json_response(self, {"error": "upload too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            _json_response(self, {"error": "expected multipart/form-data"}, HTTPStatus.BAD_REQUEST)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
        )
        try:
            job = _build_job(form)
            reconstruct_gs = _bool_field(form, "reconstruct_gs")
        except Exception as exc:  # noqa: BLE001
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        global CURRENT_JOB
        with STATE_LOCK:
            if CURRENT_JOB is not None:
                _json_response(self, {"error": "another job is already running"}, HTTPStatus.CONFLICT)
                return
            CURRENT_JOB = job
        thread = threading.Thread(target=_run_job, args=(job, reconstruct_gs), daemon=True)
        thread.start()
        _json_response(self, job.public_payload(), HTTPStatus.ACCEPTED)

    def _handle_cancel(self) -> None:
        with STATE_LOCK:
            job = CURRENT_JOB
        if job is None or job.process is None:
            _json_response(self, {"status": "idle", "message": "No running process"})
            return
        job.append_log("[ui] Cancel requested.")
        job.status = "failed"
        job.message = "Cancelled"
        try:
            job.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        _json_response(self, job.public_payload())

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Lyra 2 UI serving {ROOT} on http://{HOST}:{PORT}", flush=True)
    print(f"Job outputs: {OUTPUT_ROOT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
