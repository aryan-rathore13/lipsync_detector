"""
Extracts lip-crop frames and audio waveform from a video file.
Uses FFmpeg for A/V demux and MediaPipe FaceLandmarker (Tasks API) for per-frame lip ROI.
"""

import os
import tempfile
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import librosa

# Inner lip landmark indices (MediaPipe 478-point mesh — NOT dlib 68-point)
# Upper inner: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
# Lower inner: 78, 95, 88, 87, 14, 317, 402, 318, 324, 308
_INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      95, 88, 87, 14, 317, 402, 318, 324]

# MAR vertical pairs: inner upper → inner lower
# Pairs chosen for symmetric coverage of mouth opening
_MAR_TOP = [82, 81, 80]    # inner upper lip
_MAR_BOT = [87, 88, 95]    # inner lower lip
_MAR_LEFT = 78             # left inner corner
_MAR_RIGHT = 308           # right inner corner


def _resolve_face_landmarker_model_path(cfg: dict | None = None) -> str:
    if cfg is not None:
        configured_path = cfg.get("models", {}).get("face_landmarker")
        if configured_path:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", configured_path))
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
    )


def _make_landmarker(running_mode, model_path: str):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def extract_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Returns mono float32 waveform at `sample_rate` Hz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", str(sample_rate),
             "-vn", tmp_wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, _ = librosa.load(tmp_wav, sr=sample_rate, mono=True)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)
    return wav.astype(np.float32)


def _lip_bbox(lm, frame_w: int, frame_h: int, pad: float = 0.25):
    xs = [lm[i].x * frame_w for i in _INNER_LIP_INDICES]
    ys = [lm[i].y * frame_h for i in _INNER_LIP_INDICES]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad; x2 += bw * pad
    y1 -= bh * pad; y2 += bh * pad
    return max(0, int(x1)), max(0, int(y1)), min(frame_w, int(x2)), min(frame_h, int(y2))


def _mar(lm, frame_w: int, frame_h: int) -> float:
    """Mouth Aspect Ratio: mean vertical separation / horizontal width."""
    def pt(i):
        return np.array([lm[i].x * frame_w, lm[i].y * frame_h])
    vert = np.mean([np.linalg.norm(pt(t) - pt(b))
                    for t, b in zip(_MAR_TOP, _MAR_BOT)])
    horiz = np.linalg.norm(pt(_MAR_LEFT) - pt(_MAR_RIGHT)) + 1e-6
    return float(vert / horiz)


def extract_lip_frames(
    video_path: str,
    model_path: str,
    target_fps: int = 25,
    crop_size: int = 96,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
        lip_frames   : (N, crop_size, crop_size, 3) uint8 BGR
        mar_sequence : (N,) float32
        actual_fps   : float
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_vid = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"fps={target_fps}",
             "-an", tmp_vid],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cap = cv2.VideoCapture(tmp_vid)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)

        landmarker = _make_landmarker(
            mp_vision.RunningMode.VIDEO,
            model_path,
        )
        lip_frames, mar_sequence = [], []
        last_crop = None
        last_mar = 0.0
        frame_idx = 0
        ms_per_frame = 1000.0 / actual_fps

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * ms_per_frame)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                x1, y1, x2, y2 = _lip_bbox(lm, w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (crop_size, crop_size))
                    lip_frames.append(crop)
                    last_crop = crop
                    last_mar = _mar(lm, w, h)
                    mar_sequence.append(last_mar)
                    frame_idx += 1
                    continue

            if last_crop is not None:
                lip_frames.append(last_crop.copy())
                mar_sequence.append(last_mar)
            else:
                lip_frames.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
                mar_sequence.append(0.0)
            frame_idx += 1

        cap.release()
        landmarker.close()
    finally:
        if os.path.exists(tmp_vid):
            os.unlink(tmp_vid)

    return (
        np.array(lip_frames, dtype=np.uint8),
        np.array(mar_sequence, dtype=np.float32),
        actual_fps,
    )


def preprocess(video_path: str, cfg: dict) -> dict:
    fps = cfg["video"]["target_fps"]
    size = cfg["video"]["lip_crop_size"]
    sr = cfg["audio"]["sample_rate"]
    model_path = _resolve_face_landmarker_model_path(cfg)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Face landmarker model not found at {model_path}. "
            "Run scripts/download_models.sh or update config.yaml."
        )

    lip_frames, mar_sequence, actual_fps = extract_lip_frames(video_path, model_path, fps, size)
    audio_wav = extract_audio(video_path, sr)

    return {
        "lip_frames": lip_frames,
        "mar_sequence": mar_sequence,
        "audio_wav": audio_wav,
        "fps": actual_fps,
        "sample_rate": sr,
    }
