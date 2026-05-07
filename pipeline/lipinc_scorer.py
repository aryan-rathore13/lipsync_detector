"""
LipForensics spatiotemporal inconsistency scorer.
Uses github.com/ahaliassos/LipForensics (ResNet + MS-TCN temporal transformer).

Model outputs a single forgery logit per clip; higher = more fake.
Input: grayscale 88x88 mouth crops, 25 frames per clip.

Setup:
    # Weights must be downloaded manually from Google Drive:
    # https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7/view
    # Place at: models/lipinc_v2/lipforensics_ff.pth

Falls back to a frame-difference jitter heuristic if weights are absent.

Returns:
    lipinc_fake_score : float in [0, 1]
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import cv2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(lipforensics_repo: str):
    """Instantiate LipForensics Lipreading model using its own config."""
    cfg_path = os.path.join(lipforensics_repo, "models", "configs", "lrw_resnet18_mstcn.json")
    with open(cfg_path) as f:
        args = json.load(f)

    if lipforensics_repo not in sys.path:
        sys.path.insert(0, lipforensics_repo)

    from models.spatiotemporal_net import Lipreading  # noqa: PLC0415

    tcn_options = {
        "num_layers": args["tcn_num_layers"],
        "kernel_size": args["tcn_kernel_size"],
        "dropout": args["tcn_dropout"],
        "dwpw": args["tcn_dwpw"],
        "width_mult": args["tcn_width_mult"],
    }
    return Lipreading(num_classes=1, tcn_options=tcn_options, relu_type=args["relu_type"])


def _lipforensics_score(lip_frames: np.ndarray, weights_path: str, device) -> float:
    """
    Run LipForensics and return fake probability in [0, 1].
    Processes the video in non-overlapping 25-frame clips, averages logits, then sigmoids.
    """
    repo = os.path.join(os.path.dirname(__file__), "..", "lipforensics")
    model = _build_model(repo)

    # Load weights — map to CPU first, then move to target device
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prepare grayscale 88x88 frames
    target_size = 88
    mean, std = 0.421, 0.165

    def prep_frame(f):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_size, target_size))
        t = torch.from_numpy(gray.astype(np.float32) / 255.0)
        t = (t - mean) / std
        return t  # (H, W)

    prepped = [prep_frame(f) for f in lip_frames]

    frames_per_clip = 25
    logits_all = []

    for start in range(0, len(prepped) - frames_per_clip + 1, frames_per_clip):
        clip = prepped[start: start + frames_per_clip]
        # (1, 1, T, H, W)
        clip_t = torch.stack(clip).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(clip_t, lengths=[frames_per_clip])  # (1, 1)
        logits_all.append(logit.squeeze().cpu().float())

    if not logits_all:
        return 0.5

    mean_logit = torch.mean(torch.stack(logits_all))
    fake_prob = float(torch.sigmoid(mean_logit))
    return fake_prob


def _fallback_jitter_score(lip_frames: np.ndarray) -> float:
    """
    Heuristic: measures frame-to-frame vs smoothed-frame difference ratio.
    Generative lip regions exhibit higher short-term jitter relative to slower structural motion.
    """
    if len(lip_frames) < 10:
        return 0.5

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in lip_frames]

    short_diffs = [float(np.mean(np.abs(grays[i+1] - grays[i])))
                   for i in range(len(grays) - 1)]
    long_diffs = [float(np.mean(np.abs(grays[i] - grays[i-4])))
                  for i in range(4, len(grays))]

    if not short_diffs or not long_diffs:
        return 0.5

    short_mean = np.mean(short_diffs)
    long_mean = np.mean(long_diffs)

    if long_mean < 1e-6:
        return 0.5

    jitter_ratio = short_mean / (long_mean + 1e-6)
    # Authentic ~0.6–1.2, fake often > 1.5
    score = float(np.clip((jitter_ratio - 0.6) / 1.5, 0.0, 1.0))
    return score


def run(data: dict, cfg: dict) -> dict:
    weights = os.path.join(
        os.path.dirname(__file__), "..", cfg["models"]["lipinc_weights"]
    )
    lipforensics_repo = os.path.join(os.path.dirname(__file__), "..", "lipforensics")
    lip_frames = data["lip_frames"]
    device = get_device()

    if os.path.exists(weights) and os.path.exists(lipforensics_repo):
        try:
            score = _lipforensics_score(lip_frames, weights, device)
            return {"lipinc_fake_score": float(score), "method": "lipforensics",
                    "skipped": False}
        except Exception as e:
            print(f"      [LipForensics] error: {e} — using jitter fallback")

    return {"lipinc_fake_score": _fallback_jitter_score(lip_frames),
            "method": "jitter_fallback", "skipped": False}
