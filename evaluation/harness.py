"""
Batch evaluation harness for the lip-sync detector.

Expected manifest CSV columns:
    video_path,label

Optional columns:
    sample_id,split,language,source,notes

Label values accepted:
    REAL, FAKE, 0, 1, true, false
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from detector import detect, load_config
from evaluation.reporting import generate_report


REQUIRED_COLUMNS = {"video_path", "label"}
OPTIONAL_GROUP_FIELDS = ("split", "language", "source")


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    video_path: str
    label: int
    metadata: dict


def _parse_label(raw: str) -> int:
    value = str(raw).strip().lower()
    if value in {"1", "fake", "true", "yes"}:
        return 1
    if value in {"0", "real", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported label value: {raw}")


def load_manifest(manifest_path: str) -> list[EvalSample]:
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV is missing a header row.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        samples = []
        for idx, row in enumerate(reader, start=1):
            raw_video_path = (row.get("video_path") or "").strip()
            if not raw_video_path:
                continue
            video_path = raw_video_path
            if not os.path.isabs(video_path):
                video_path = os.path.abspath(os.path.join(manifest_dir, video_path))
            sample_id = row.get("sample_id") or f"sample_{idx:04d}"
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    video_path=video_path,
                    label=_parse_label(row["label"]),
                    metadata={
                        key: value.strip()
                        for key, value in row.items()
                        if key not in {"sample_id", "video_path", "label"} and value is not None
                    },
                )
            )

    if not samples:
        raise ValueError("Manifest is empty.")

    return samples


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _confusion_counts(labels: list[int], predictions: list[int]) -> dict:
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    predictions = [1 if score >= threshold else 0 for score in scores]
    counts = _confusion_counts(labels, predictions)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(labels))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "threshold": threshold,
        "support": len(labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": counts,
    }


def compute_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None

    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5

    return wins / (len(positives) * len(negatives))


def threshold_sweep(
    labels: list[int],
    scores: list[float],
    step: float = 0.05,
) -> list[dict]:
    thresholds = []
    current = 0.0
    while current < 1.0:
        thresholds.append(round(current, 10))
        current += step
    thresholds.append(1.0)

    return [compute_metrics(labels, scores, threshold=value) for value in thresholds]


def best_threshold(sweep_rows: list[dict], metric: str) -> dict:
    return max(
        sweep_rows,
        key=lambda row: (
            row[metric],
            row["accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )


def summarize_groups(
    predictions: list[dict],
    group_field: str,
    threshold: float,
) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in predictions:
        group_value = row.get(group_field)
        if not group_value:
            continue
        grouped.setdefault(group_value, []).append(row)

    summary = {}
    for group_value, rows in grouped.items():
        labels = [row["label"] for row in rows]
        scores = [row["confidence"] for row in rows]
        metrics = compute_metrics(labels, scores, threshold)
        metrics["roc_auc"] = compute_roc_auc(labels, scores)
        summary[group_value] = metrics

    return summary


def evaluate_manifest(
    manifest_path: str,
    config_path: str,
    output_dir: str,
    threshold_step: float = 0.05,
    continue_on_error: bool = True,
    detect_fn: Callable[[str, dict, bool], dict] = detect,
) -> dict:
    samples = load_manifest(manifest_path)
    cfg = load_config(config_path)
    operating_threshold = float(cfg["thresholds"]["final_verdict"])

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    predictions = []
    failures = []
    t0 = time.time()

    for index, sample in enumerate(samples, start=1):
        row = {
            "sample_id": sample.sample_id,
            "video_path": sample.video_path,
            "label": sample.label,
            **sample.metadata,
        }
        try:
            result = detect_fn(sample.video_path, cfg, False)
            row.update(
                {
                    "status": "ok",
                    "verdict": result.get("verdict"),
                    "predicted_label": 1 if result.get("confidence", 0.0) >= operating_threshold else 0,
                    "confidence": float(result.get("confidence", 0.0)),
                    "triggered_by": result.get("triggered_by"),
                    "processing_time_s": result.get("processing_time_s"),
                    "syncnet_lse_c": result.get("syncnet_lse_c"),
                    "syncnet_lse_d": result.get("syncnet_lse_d"),
                    "syncnet_offset_frames": result.get("syncnet_offset_frames"),
                    "lipinc_method": result.get("lipinc_method"),
                    "biolip_method": result.get("biolip_method"),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "verdict": None,
                    "predicted_label": None,
                    "confidence": math.nan,
                    "triggered_by": None,
                    "processing_time_s": None,
                }
            )
            failures.append(row)
            if not continue_on_error:
                raise

        predictions.append(row)
        print(f"[{index}/{len(samples)}] {sample.sample_id} -> {row['status']}")

    successful = [row for row in predictions if row["status"] == "ok"]
    labels = [row["label"] for row in successful]
    scores = [row["confidence"] for row in successful]

    metrics = None
    sweep_rows = []
    best_f1 = None
    best_balanced_accuracy = None
    roc_auc = None
    group_summaries = {}

    if successful:
        metrics = compute_metrics(labels, scores, operating_threshold)
        roc_auc = compute_roc_auc(labels, scores)
        sweep_rows = threshold_sweep(labels, scores, step=threshold_step)
        best_f1 = best_threshold(sweep_rows, "f1")
        best_balanced_accuracy = best_threshold(sweep_rows, "balanced_accuracy")
        for field in OPTIONAL_GROUP_FIELDS:
            group_summaries[field] = summarize_groups(successful, field, operating_threshold)

    triggered_breakdown = {}
    for row in successful:
        key = row.get("triggered_by") or "unknown"
        triggered_breakdown[key] = triggered_breakdown.get(key, 0) + 1

    summary = {
        "manifest_path": os.path.abspath(manifest_path),
        "config_path": os.path.abspath(config_path),
        "run_dir": os.path.abspath(run_dir),
        "num_samples": len(samples),
        "num_successful": len(successful),
        "num_failed": len(failures),
        "operating_threshold": operating_threshold,
        "threshold_step": threshold_step,
        "elapsed_s": round(time.time() - t0, 2),
        "mean_processing_time_s": (
            round(statistics.mean(row["processing_time_s"] for row in successful), 3)
            if successful
            else None
        ),
        "roc_auc": roc_auc,
        "metrics_at_operating_threshold": metrics,
        "best_f1_threshold": best_f1,
        "best_balanced_accuracy_threshold": best_balanced_accuracy,
        "triggered_by_breakdown": triggered_breakdown,
        "group_summaries": group_summaries,
        "failures": failures,
    }

    _write_predictions(predictions, os.path.join(run_dir, "predictions.jsonl"))
    _write_csv(predictions, os.path.join(run_dir, "predictions.csv"))
    _write_csv(sweep_rows, os.path.join(run_dir, "threshold_sweep.csv"))
    with open(os.path.join(run_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as handle:
        json.dump(cfg, handle, indent=2)

    report_artifacts = generate_report(run_dir)
    summary["report_path"] = report_artifacts["report_path"]
    summary["plot_paths"] = report_artifacts["plot_paths"]
    with open(os.path.join(run_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _write_predictions(rows: list[dict], path: str) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
