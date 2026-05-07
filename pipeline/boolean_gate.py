"""
Fast boolean pre-filter.
Detects trivially obvious fakes:
  - audio active but lips stationary
  - lips moving but audio silent
Returns a score in [0, 1] and an instant_verdict flag.
"""

import numpy as np
import librosa


def _audio_activity_per_window(
    audio_wav: np.ndarray,
    sample_rate: int,
    window_sec: float,
    rms_threshold: float,
) -> np.ndarray:
    """Bool array: True for each 1s window where audio RMS exceeds threshold."""
    hop = int(window_sec * sample_rate)
    if hop <= 0 or len(audio_wav) < hop:
        return np.array([], dtype=bool)
    rms = librosa.feature.rms(
        y=audio_wav,
        frame_length=hop,
        hop_length=hop,
        center=False,
    )[0]
    return rms > rms_threshold


def _lip_activity_per_window(
    mar_sequence: np.ndarray,
    fps: float,
    window_sec: float,
    var_threshold: float,
) -> np.ndarray:
    """Bool array: True for each 1s window where MAR variance exceeds threshold."""
    window_frames = max(1, int(window_sec * fps))
    n_windows = len(mar_sequence) // window_frames
    activity = []
    for i in range(n_windows):
        chunk = mar_sequence[i * window_frames: (i + 1) * window_frames]
        activity.append(float(np.var(chunk)) > var_threshold)
    return np.array(activity, dtype=bool)


def run(data: dict, cfg: dict) -> dict:
    """
    Returns:
        score           : float  1.0 = definitely fake, 0.0 = inconclusive
        triggered       : bool   True if gate fires (instant verdict)
        mismatch_ratio  : float  fraction of windows with audio/lip mismatch
        details         : dict   per-window arrays for debugging
    """
    audio_wav = data["audio_wav"]
    mar_sequence = data["mar_sequence"]
    fps = data["fps"]
    sr = data["sample_rate"]
    t = cfg["thresholds"]

    audio_active = _audio_activity_per_window(
        audio_wav, sr, window_sec=1.0, rms_threshold=t["rms_energy"]
    )
    lip_active = _lip_activity_per_window(
        mar_sequence, fps, window_sec=1.0, var_threshold=t["mar_variance"]
    )

    # Align to shorter array length
    n = min(len(audio_active), len(lip_active))
    audio_active = audio_active[:n]
    lip_active = lip_active[:n]

    if n == 0:
        return {"score": 0.0, "triggered": False, "mismatch_ratio": 0.0,
                "details": {}}

    mismatch = audio_active ^ lip_active  # XOR: one active, other silent
    mismatch_ratio = float(np.mean(mismatch))
    triggered = mismatch_ratio >= t["boolean_gate_mismatch_ratio"]
    score = mismatch_ratio if triggered else 0.0

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "triggered": triggered,
        "mismatch_ratio": mismatch_ratio,
        "details": {
            "audio_active": audio_active.tolist(),
            "lip_active": lip_active.tolist(),
            "mismatch": mismatch.tolist(),
        },
    }
