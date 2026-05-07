"""
Threshold calibration helpers built on top of evaluation outputs.
"""

from __future__ import annotations

import copy
import json
import os

from detector import load_config


SUPPORTED_METRICS = {"f1", "balanced_accuracy", "accuracy", "precision", "recall", "specificity"}


def load_summary(run_dir: str) -> dict:
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path) as handle:
        return json.load(handle)


def select_threshold(summary: dict, metric: str = "f1") -> dict:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of: {sorted(SUPPORTED_METRICS)}")

    if metric == "f1":
        row = summary.get("best_f1_threshold")
    elif metric == "balanced_accuracy":
        row = summary.get("best_balanced_accuracy_threshold")
    else:
        sweep_rows = summary.get("threshold_sweep") or []
        if not sweep_rows:
            raise ValueError("Summary does not include threshold sweep data.")
        row = max(
            sweep_rows,
            key=lambda entry: (
                entry.get(metric, 0.0),
                entry.get("accuracy", 0.0),
                -abs(entry.get("threshold", 0.5) - 0.5),
            ),
        )

    if not row:
        raise ValueError(f"No threshold candidate found for metric '{metric}'.")
    return row


def build_tuned_config(config_path: str, threshold: float) -> dict:
    cfg = load_config(config_path)
    tuned = copy.deepcopy(cfg)
    tuned.setdefault("thresholds", {})
    tuned["thresholds"]["final_verdict"] = float(threshold)
    return tuned


def write_tuned_config(config: dict, output_path: str) -> str:
    import yaml

    with open(output_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path
