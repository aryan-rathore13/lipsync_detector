"""
Generate markdown and plot artifacts from an evaluation run directory.
"""

from __future__ import annotations

import csv
import json
import os


def _load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_ratio(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_seconds(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}s"


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_threshold_sweep(sweep_rows: list[dict], output_path: str) -> str | None:
    if not sweep_rows:
        return None

    plt = _import_matplotlib()
    thresholds = [_to_float(row["threshold"]) for row in sweep_rows]
    f1_scores = [_to_float(row["f1"]) for row in sweep_rows]
    balanced_accuracy = [_to_float(row["balanced_accuracy"]) for row in sweep_rows]
    accuracy = [_to_float(row["accuracy"]) for row in sweep_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, f1_scores, label="F1", linewidth=2.0)
    ax.plot(thresholds, balanced_accuracy, label="Balanced Accuracy", linewidth=2.0)
    ax.plot(thresholds, accuracy, label="Accuracy", linewidth=2.0)
    ax.set_title("Threshold Sweep")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_score_distribution(predictions: list[dict], output_path: str) -> str | None:
    successful = [row for row in predictions if row.get("status") == "ok"]
    if not successful:
        return None

    real_scores = [_to_float(row.get("confidence")) for row in successful if int(row["label"]) == 0]
    fake_scores = [_to_float(row.get("confidence")) for row in successful if int(row["label"]) == 1]
    if not real_scores and not fake_scores:
        return None

    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [i / 20.0 for i in range(21)]
    if real_scores:
        ax.hist(real_scores, bins=bins, alpha=0.65, label="REAL", color="#4C78A8")
    if fake_scores:
        ax.hist(fake_scores, bins=bins, alpha=0.65, label="FAKE", color="#E45756")
    ax.set_title("Confidence Distribution by Label")
    ax.set_xlabel("Predicted Fake Confidence")
    ax.set_ylabel("Sample Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_trigger_breakdown(summary: dict, output_path: str) -> str | None:
    breakdown = summary.get("triggered_by_breakdown") or {}
    if not breakdown:
        return None

    plt = _import_matplotlib()
    labels = list(breakdown.keys())
    counts = [int(breakdown[key]) for key in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, counts, color=["#72B7B2", "#F58518", "#54A24B", "#EECA3B"][: len(labels)])
    ax.set_title("Detector Decision Breakdown")
    ax.set_xlabel("Triggered By")
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _metrics_table(metrics: dict | None) -> str:
    if not metrics:
        return "_No successful samples available._"

    confusion = metrics.get("confusion_matrix", {})
    rows = [
        ("Accuracy", _fmt_ratio(metrics.get("accuracy"))),
        ("Precision", _fmt_ratio(metrics.get("precision"))),
        ("Recall", _fmt_ratio(metrics.get("recall"))),
        ("F1", _fmt_ratio(metrics.get("f1"))),
        ("Specificity", _fmt_ratio(metrics.get("specificity"))),
        ("Balanced Accuracy", _fmt_ratio(metrics.get("balanced_accuracy"))),
        (
            "Confusion Matrix",
            f"TP={confusion.get('tp', 0)} TN={confusion.get('tn', 0)} FP={confusion.get('fp', 0)} FN={confusion.get('fn', 0)}",
        ),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _best_threshold_table(summary: dict) -> str:
    candidates = [
        ("Best F1", summary.get("best_f1_threshold")),
        ("Best Balanced Accuracy", summary.get("best_balanced_accuracy_threshold")),
    ]
    lines = ["| Selection Rule | Threshold | Accuracy | F1 | Balanced Accuracy |", "|---|---:|---:|---:|---:|"]
    has_row = False
    for label, row in candidates:
        if not row:
            continue
        has_row = True
        lines.append(
            f"| {label} | {_fmt_ratio(row.get('threshold'))} | {_fmt_ratio(row.get('accuracy'))} | "
            f"{_fmt_ratio(row.get('f1'))} | {_fmt_ratio(row.get('balanced_accuracy'))} |"
        )
    return "\n".join(lines) if has_row else "_No threshold sweep data available._"


def _group_table(group_name: str, group_rows: dict) -> str:
    if not group_rows:
        return f"### By {group_name}\n\n_No group data available._"

    lines = [
        f"### By {group_name}",
        "",
        "| Group | Support | Accuracy | F1 | Balanced Accuracy | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_value, metrics in sorted(group_rows.items()):
        lines.append(
            f"| {group_value} | {metrics.get('support', 0)} | {_fmt_ratio(metrics.get('accuracy'))} | "
            f"{_fmt_ratio(metrics.get('f1'))} | {_fmt_ratio(metrics.get('balanced_accuracy'))} | "
            f"{_fmt_ratio(metrics.get('roc_auc')) if metrics.get('roc_auc') is not None else 'n/a'} |"
        )
    return "\n".join(lines)


def build_markdown(summary: dict, predictions: list[dict], plot_paths: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Run Overview",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Manifest | `{summary.get('manifest_path', 'n/a')}` |",
        f"| Config | `{summary.get('config_path', 'n/a')}` |",
        f"| Samples | {summary.get('num_samples', 0)} |",
        f"| Successful | {summary.get('num_successful', 0)} |",
        f"| Failed | {summary.get('num_failed', 0)} |",
        f"| Operating Threshold | {_fmt_ratio(summary.get('operating_threshold'))} |",
        f"| ROC-AUC | {_fmt_ratio(summary.get('roc_auc')) if summary.get('roc_auc') is not None else 'n/a'} |",
        f"| Mean Processing Time | {_fmt_seconds(summary.get('mean_processing_time_s'))} |",
        "",
        "## Metrics At Operating Threshold",
        "",
        _metrics_table(summary.get("metrics_at_operating_threshold")),
        "",
        "## Best Threshold Candidates",
        "",
        _best_threshold_table(summary),
        "",
    ]

    if plot_paths.get("threshold_sweep"):
        lines.extend(["## Threshold Sweep Plot", "", f"![Threshold Sweep]({os.path.basename(plot_paths['threshold_sweep'])})", ""])
    if plot_paths.get("score_distribution"):
        lines.extend(["## Confidence Distribution", "", f"![Confidence Distribution]({os.path.basename(plot_paths['score_distribution'])})", ""])
    if plot_paths.get("trigger_breakdown"):
        lines.extend(["## Decision Breakdown", "", f"![Decision Breakdown]({os.path.basename(plot_paths['trigger_breakdown'])})", ""])

    group_summaries = summary.get("group_summaries", {})
    for group_name, rows in group_summaries.items():
        lines.extend([_group_table(group_name, rows), ""])

    failures = summary.get("failures") or []
    if failures:
        lines.extend(
            [
                "## Failures",
                "",
                "| Sample ID | Video Path | Error |",
                "|---|---|---|",
            ]
        )
        for row in failures:
            lines.append(f"| {row.get('sample_id', 'n/a')} | `{row.get('video_path', 'n/a')}` | {row.get('error', 'n/a')} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Predictions are saved in both JSONL and CSV for downstream analysis.",
            "- Threshold plots are generated from `threshold_sweep.csv`.",
            "- Group tables only appear when the manifest includes fields such as `split`, `language`, or `source`.",
        ]
    )
    return "\n".join(lines)


def generate_report(run_dir: str) -> dict:
    summary_path = os.path.join(run_dir, "summary.json")
    predictions_path = os.path.join(run_dir, "predictions.csv")
    sweep_path = os.path.join(run_dir, "threshold_sweep.csv")

    summary = _load_json(summary_path)
    predictions = _load_csv(predictions_path)
    sweep_rows = _load_csv(sweep_path)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_paths = {
        "threshold_sweep": plot_threshold_sweep(sweep_rows, os.path.join(plot_dir, "threshold_sweep.png")),
        "score_distribution": plot_score_distribution(predictions, os.path.join(plot_dir, "score_distribution.png")),
        "trigger_breakdown": plot_trigger_breakdown(summary, os.path.join(plot_dir, "trigger_breakdown.png")),
    }

    markdown = build_markdown(summary, predictions, plot_paths)
    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as handle:
        handle.write(markdown)

    return {
        "report_path": report_path,
        "plot_paths": {key: value for key, value in plot_paths.items() if value},
    }
