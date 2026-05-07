"""
Biomechanical lip kinematic scorer.

Priority:
  1. BioLip pretrained model if available.
  2. Custom kinematic jitter detector using MediaPipe landmark velocities
     and accelerations — flags frames that violate human muscle-contraction
     physics (acceleration spikes, unnatural velocity reversals).

Returns:
    biolip_fake_score : float in [0, 1]
"""

import os
import sys
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2

_LIP_KIN_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                    95, 88, 87, 14, 317, 402, 318, 324]


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


def _extract_lip_coordinates(lip_frames: np.ndarray, model_path: str) -> np.ndarray:
    """
    Re-runs MediaPipe on lip crop frames to get precise landmark coords.
    Returns (N, K, 2) float32 array of (x, y) pixel coords for K landmarks.
    """
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
    coords = []
    h, w = lip_frames.shape[1], lip_frames.shape[2]

    for frame in lip_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            pts = np.array(
                [[lm[i].x * w, lm[i].y * h] for i in _LIP_KIN_INDICES],
                dtype=np.float32
            )
        else:
            pts = np.zeros((len(_LIP_KIN_INDICES), 2), dtype=np.float32)
        coords.append(pts)

    landmarker.close()
    return np.array(coords, dtype=np.float32)  # (N, K, 2)


def _kinematic_violation_score(coords: np.ndarray, fps: float) -> float:
    """
    Computes fraction of frames with biomechanically implausible lip motion.

    Human lip muscle maximum acceleration is bounded by soft tissue physics.
    Deepfake generators produce frame sequences with acceleration spikes that
    exceed these physical limits.

    Thresholds derived from BioLip paper (normalised pixel units at 96px crop):
      velocity_max    : 8 px/frame   (soft tissue speed limit)
      acceleration_max: 6 px/frame²  (muscle contraction limit)
    """
    if len(coords) < 3:
        return 0.5

    # Per-frame mean displacement across all tracked landmarks
    displacements = np.linalg.norm(
        coords[1:] - coords[:-1], axis=-1
    ).mean(axis=-1)  # (N-1,)

    velocity = displacements  # px/frame
    acceleration = np.abs(np.diff(velocity))  # px/frame²

    vel_max = 8.0   # px/frame at 96px crop, 25fps
    acc_max = 6.0   # px/frame²

    vel_violations = np.sum(velocity > vel_max)
    acc_violations = np.sum(acceleration > acc_max)

    n = len(velocity) + len(acceleration)
    if n == 0:
        return 0.5

    violation_rate = (vel_violations + acc_violations) / n
    return float(np.clip(violation_rate * 2.0, 0.0, 1.0))


def _biolip_model_score(lip_frames: np.ndarray, weights_path: str, device) -> float:
    """Run pretrained BioLip model if available."""
    repo = os.path.join(os.path.dirname(__file__), "..", "biolip")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # BioLip inference will be wired here once repo is confirmed public
    raise NotImplementedError("BioLip repo not yet integrated — using kinematic fallback")


def run(data: dict, cfg: dict) -> dict:
    weights = os.path.join(
        os.path.dirname(__file__), "..", cfg["models"]["biolip_weights"]
    )
    lip_frames = data["lip_frames"]
    fps = data["fps"]
    device = get_device()
    face_landmarker_path = _resolve_face_landmarker_model_path(cfg)

    biolip_repo = os.path.join(os.path.dirname(__file__), "..", "biolip")

    if os.path.exists(weights) and os.path.exists(biolip_repo):
        try:
            score = _biolip_model_score(lip_frames, weights, device)
            return {"biolip_fake_score": float(score), "method": "biolip_model",
                    "skipped": False}
        except NotImplementedError:
            pass  # fall through to kinematic fallback
        except Exception:
            pass

    # Kinematic fallback: re-run MediaPipe on crops to get precise coords
    coords = _extract_lip_coordinates(lip_frames, face_landmarker_path)
    score = _kinematic_violation_score(coords, fps)

    return {"biolip_fake_score": float(score), "method": "kinematic_fallback",
            "skipped": False}
