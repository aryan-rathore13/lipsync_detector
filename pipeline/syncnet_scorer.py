"""
SyncNet-based audio-visual temporal sync scorer.
Wraps github.com/joonson/syncnet_python.

SyncNetInstance.evaluate() handles its own A/V extraction via FFmpeg
and returns (offset, confidence, dists_npy) where:
  - confidence = median_dist - min_dist  (higher = more in sync, typically 1–10)
  - offset     = frame offset found (0 = perfectly in sync)

Setup:
    cd lipsync_detector
    git clone https://github.com/joonson/syncnet_python.git syncnet_python
    wget -P models/syncnet/ http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model

Returns:
    syncnet_fake_score : float in [0, 1]  (0 = in sync, 1 = clearly out of sync)
    lse_c              : float  raw confidence (higher = more in sync)
    lse_d              : float  minimum distance (lower = more in sync)
    offset_frames      : int
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_face_landmarker_model_path(cfg: dict | None = None) -> str:
    if cfg is not None:
        configured_path = cfg.get("models", {}).get("face_landmarker")
        if configured_path:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", configured_path))
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
    )


class _Opts:
    """Minimal opts object for SyncNetInstance.evaluate()."""
    def __init__(self, tmp_dir: str, reference: str = "video", batch_size: int = 20, vshift: int = 15):
        self.tmp_dir = tmp_dir
        self.reference = reference
        self.batch_size = batch_size
        self.vshift = vshift


def _get_face_bbox(video_path: str, model_path: str) -> tuple[int, int, int, int] | None:
    """
    Sample up to 30 frames evenly, run MediaPipe, return the median face
    bounding box as (x1, y1, x2, y2) in pixels. Returns None if no face found.
    """
    import mediapipe as mp

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_indices = set(int(i * total / 30) for i in range(30))

    bboxes = []
    for idx in sorted(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            xs = [l.x for l in lm]
            ys = [l.y for l in lm]
            pad = 0.15
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            bboxes.append((
                max(0.0, min(xs) - bw * pad),
                max(0.0, min(ys) - bh * pad),
                min(1.0, max(xs) + bw * pad),
                min(1.0, max(ys) + bh * pad),
            ))
    cap.release()
    landmarker.close()

    if not bboxes:
        return None

    # Median bounding box (normalised)
    bboxes = np.array(bboxes)
    nx1, ny1, nx2, ny2 = np.median(bboxes, axis=0)

    # Read one frame to get pixel dimensions
    cap2 = cv2.VideoCapture(video_path)
    ok, frame = cap2.read()
    cap2.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return int(nx1 * w), int(ny1 * h), int(nx2 * w), int(ny2 * h)


def _make_face_crop_video(
    video_path: str,
    out_path: str,
    model_path: str,
    target_size: int = 224,
) -> None:
    """
    Uses FFmpeg to resample to 25fps, crop to the detected face bbox, and scale to
    target_size×target_size. The 25fps resample is mandatory: SyncNet hardcodes 25fps
    in its audio alignment (audio_frame = vframe * 40ms). Passing 30fps video without
    resampling causes a cumulative ~6.7ms/frame drift that collapses LSE-C on real videos.
    """
    import subprocess as _sp

    bbox = _get_face_bbox(video_path, model_path)
    if bbox:
        x1, y1, x2, y2 = bbox
        cw = max(1, x2 - x1)
        ch = max(1, y2 - y1)
        side = max(cw, ch)
        cx = x1 + cw // 2 - side // 2
        cy = y1 + ch // 2 - side // 2
        vf = f"fps=25,crop={side}:{side}:{max(0,cx)}:{max(0,cy)},scale={target_size}:{target_size}"
    else:
        vf = f"fps=25,scale={target_size}:{target_size}:force_original_aspect_ratio=increase,crop={target_size}:{target_size}"

    _sp.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", video_path,
         "-vf", vf,
         "-an",
         "-c:v", "libx264", "-preset", "fast",
         out_path],
        check=True,
    )


def run(data: dict, cfg: dict, video_path: str) -> dict:
    """
    data       : preprocessed data dict
    cfg        : config dict
    video_path : original video file path
    """
    weights = os.path.join(
        os.path.dirname(__file__), "..", cfg["models"]["syncnet_weights"]
    )
    face_landmarker_path = _resolve_face_landmarker_model_path(cfg)
    if not os.path.exists(weights):
        return {
            "syncnet_fake_score": 0.5,
            "lse_c": 5.0, "lse_d": 5.0, "offset_frames": 0,
            "skipped": True,
            "reason": f"Model weights not found at {weights}. "
                      f"Download with: curl -L -o models/syncnet/syncnet_v2.model "
                      f"http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
        }
    if not os.path.exists(face_landmarker_path):
        return {
            "syncnet_fake_score": 0.5,
            "lse_c": 5.0, "lse_d": 5.0, "offset_frames": 0,
            "skipped": True,
            "reason": f"Face landmarker model not found at {face_landmarker_path}. "
                      "Run scripts/download_models.sh or update config.yaml.",
        }

    repo_path = os.path.join(os.path.dirname(__file__), "..", "syncnet_python")
    if not os.path.exists(repo_path):
        return {
            "syncnet_fake_score": 0.5,
            "lse_c": 5.0, "lse_d": 5.0, "offset_frames": 0,
            "skipped": True,
            "reason": "syncnet_python repo not found. "
                      "Clone with: git clone https://github.com/joonson/syncnet_python.git syncnet_python",
        }

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from SyncNetInstance import SyncNetInstance  # noqa: PLC0415
    except ImportError as e:
        return {"syncnet_fake_score": 0.5, "lse_c": 5.0, "lse_d": 5.0,
                "offset_frames": 0, "skipped": True, "reason": str(e)}

    device = get_device()
    model = SyncNetInstance(device=str(device))
    model.loadParameters(weights)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write 224×224 face-cropped video (video only, no audio)
        face_vid_silent = os.path.join(tmp_dir, "face_crop_silent.mp4")
        _make_face_crop_video(
            video_path,
            face_vid_silent,
            model_path=face_landmarker_path,
            target_size=224,
        )

        # Mux original audio back in — SyncNet extracts audio from the video file
        import subprocess as _sp
        face_vid = os.path.join(tmp_dir, "face_crop.mp4")
        _sp.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", face_vid_silent,
             "-i", video_path,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac",
             "-shortest",
             face_vid],
            check=True,
        )

        opt = _Opts(tmp_dir=tmp_dir, reference="ref", batch_size=20, vshift=15)
        try:
            offset_np, conf_np, dists_npy = model.evaluate(opt, face_vid)
        except Exception as e:
            return {"syncnet_fake_score": 0.5, "lse_c": 5.0, "lse_d": 5.0,
                    "offset_frames": 0, "skipped": True, "reason": str(e)}

    offset_frames = int(offset_np)
    lse_c = float(conf_np)                        # higher = more in sync
    lse_d = float(np.min(dists_npy.mean(axis=0))) # mean over frames, then min

    # Normalize: low confidence (fake) → high fake_score
    # Paper: authentic 3–10, fake 1–2.5; map linearly [1,10] → [1,0]
    syncnet_fake_score = float(np.clip(1.0 - (lse_c - 1.0) / 9.0, 0.0, 1.0))

    return {
        "syncnet_fake_score": syncnet_fake_score,
        "lse_c": lse_c,
        "lse_d": lse_d,
        "offset_frames": offset_frames,
        "skipped": False,
    }
