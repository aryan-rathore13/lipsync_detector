#!/usr/bin/env python3

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.calibration import build_tuned_config, load_summary, select_threshold, write_tuned_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a tuned decision threshold from an evaluation run.")
    parser.add_argument("run_dir", help="Evaluation run directory containing summary.json")
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "balanced_accuracy", "accuracy", "precision", "recall", "specificity"],
        help="Metric used to select the threshold",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "config.yaml"),
        help="Base config to update",
    )
    parser.add_argument(
        "--write-config",
        help="Optional output path for a tuned config file",
    )
    args = parser.parse_args()

    summary = load_summary(args.run_dir)
    selected = select_threshold(summary, metric=args.metric)

    response = {
        "run_dir": os.path.abspath(args.run_dir),
        "metric": args.metric,
        "selected_threshold": selected,
    }

    if args.write_config:
        tuned = build_tuned_config(args.config, selected["threshold"])
        output_path = write_tuned_config(tuned, args.write_config)
        response["written_config"] = os.path.abspath(output_path)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
