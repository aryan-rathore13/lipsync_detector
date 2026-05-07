"""
Fuses scores from all pipeline layers into a single FAKE/REAL verdict with confidence.
"""

import numpy as np


def fuse(
    boolean_result: dict,
    syncnet_result: dict,
    lipinc_result: dict,
    biolip_result: dict,
    cfg: dict,
) -> dict:
    """
    Returns final verdict dict:
      verdict      : "FAKE" | "REAL"
      confidence   : float [0.0 = definitely real, 1.0 = definitely fake]
      layer_scores : per-layer raw scores
    """
    weights = cfg["weights"]
    threshold = cfg["thresholds"]["final_verdict"]

    # Short-circuit: if boolean gate fired, it's a definitive fake
    if boolean_result.get("triggered", False):
        return {
            "verdict": "FAKE",
            "confidence": 1.0,
            "triggered_by": "boolean_gate",
            "layer_scores": {
                "boolean_gate": boolean_result["score"],
                "syncnet": None,
                "lipinc": None,
                "biolip": None,
            },
        }

    # Gather scores from each layer
    scores = {}

    if not syncnet_result.get("skipped", False):
        scores["syncnet"] = syncnet_result["syncnet_fake_score"]
    else:
        scores["syncnet"] = None

    scores["lipinc"] = lipinc_result["lipinc_fake_score"]
    scores["biolip"] = biolip_result["biolip_fake_score"]

    # SyncNet veto: fire only when LSE-C is below the configured threshold.
    # Default is 0.5 (extreme desync only). The original paper threshold of 2.5
    # is too aggressive for non-BBC / non-English / Indian-accented content where
    # authentic videos naturally score lower.
    syncnet_veto_threshold = cfg["thresholds"].get("syncnet_veto", 0.5)
    lse_c = syncnet_result.get("lse_c")
    if lse_c is not None and lse_c < syncnet_veto_threshold:
        return {
            "verdict": "FAKE",
            "confidence": float(np.clip(1.0 - lse_c / 2.5, 0.0, 1.0)),
            "triggered_by": "syncnet_veto",
            "layer_scores": {
                "boolean_gate": boolean_result.get("mismatch_ratio", 0.0),
                "syncnet": scores.get("syncnet"),
                "lipinc": scores.get("lipinc"),
                "biolip": scores.get("biolip"),
            },
            "syncnet_lse_c": lse_c,
            "syncnet_lse_d": syncnet_result.get("lse_d"),
            "syncnet_offset_frames": syncnet_result.get("offset_frames"),
            "lipinc_method": lipinc_result.get("method"),
            "biolip_method": biolip_result.get("method"),
        }

    # Weighted average (skip None layers, redistribute weight proportionally)
    active = {k: v for k, v in scores.items() if v is not None}
    if not active:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.5,
            "layer_scores": scores,
        }

    total_weight = sum(weights[k] for k in active)
    weighted_sum = sum(weights[k] * active[k] for k in active)
    confidence = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
    verdict = "FAKE" if confidence >= threshold else "REAL"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "triggered_by": "ensemble",
        "layer_scores": {
            "boolean_gate": boolean_result.get("mismatch_ratio", 0.0),
            "syncnet": scores.get("syncnet"),
            "lipinc": scores.get("lipinc"),
            "biolip": scores.get("biolip"),
        },
        "syncnet_lse_c": syncnet_result.get("lse_c"),
        "syncnet_lse_d": syncnet_result.get("lse_d"),
        "syncnet_offset_frames": syncnet_result.get("offset_frames"),
        "lipinc_method": lipinc_result.get("method"),
        "biolip_method": biolip_result.get("method"),
    }
