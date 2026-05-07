from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from detector import detect, load_config


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
ALLOWED_UPLOAD_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DetectByPathRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or repo-local path to a video file")


class DetectionSource(BaseModel):
    mode: str
    video_path: str | None = None
    filename: str | None = None


class DetectionResponse(BaseModel):
    request_id: str
    config_path: str
    source: DetectionSource
    result: dict[str, Any]


def _validate_video_path(video_path: str) -> str:
    resolved = os.path.abspath(video_path)
    if not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail=f"Video file not found: {resolved}")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=400, detail=f"Path is not a file: {resolved}")
    return resolved


def _validate_upload_name(filename: str | None) -> str:
    safe_name = filename or "upload.mp4"
    suffix = os.path.splitext(safe_name)[1].lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}")
    return safe_name


def _load_runtime_config(config_loader: Callable[[str], dict], config_path: str) -> dict:
    resolved = os.path.abspath(config_path)
    if not os.path.exists(resolved):
        raise HTTPException(status_code=500, detail=f"Config file not found: {resolved}")
    return config_loader(resolved)


def create_app(
    config_path: str | None = None,
    detect_fn: Callable[[str, dict, bool], dict] = detect,
    config_loader: Callable[[str], dict] = load_config,
) -> FastAPI:
    resolved_config_path = os.path.abspath(config_path or DEFAULT_CONFIG_PATH)
    app = FastAPI(
        title="Lip-Sync Detector API",
        version="1.0.0",
        description="HTTP service wrapper for the lip-sync deepfake detector.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "config_path": resolved_config_path,
            "config_exists": os.path.exists(resolved_config_path),
        }

    @app.post("/detect/path", response_model=DetectionResponse)
    def detect_by_path(payload: DetectByPathRequest) -> DetectionResponse:
        runtime_config = _load_runtime_config(config_loader, resolved_config_path)
        video_path = _validate_video_path(payload.video_path)
        request_id = str(uuid.uuid4())

        try:
            result = detect_fn(video_path, runtime_config, False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Detection failed: {type(exc).__name__}: {exc}") from exc

        return DetectionResponse(
            request_id=request_id,
            config_path=resolved_config_path,
            source=DetectionSource(mode="path", video_path=video_path),
            result=result,
        )

    @app.post("/detect/upload", response_model=DetectionResponse)
    async def detect_by_upload(file: UploadFile = File(...)) -> DetectionResponse:
        runtime_config = _load_runtime_config(config_loader, resolved_config_path)
        safe_name = _validate_upload_name(file.filename)
        suffix = os.path.splitext(safe_name)[1].lower()
        request_id = str(uuid.uuid4())

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            tmp_path = handle.name
            content = await file.read()
            handle.write(content)

        try:
            result = detect_fn(tmp_path, runtime_config, False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Detection failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return DetectionResponse(
            request_id=request_id,
            config_path=resolved_config_path,
            source=DetectionSource(mode="upload", filename=safe_name),
            result=result,
        )

    return app


app = create_app()
