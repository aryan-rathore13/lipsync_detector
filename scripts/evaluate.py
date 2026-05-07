#!/usr/bin/env python3

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.harness import evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch evaluation for the lip-sync detector.")
    parser.add_argument("manifest", help="CSV manifest with at least video_path,label columns")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "results", "eval"),
        help="Directory where evaluation artifacts will be saved",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold increment used for the sweep between 0.0 and 1.0",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first sample error instead of continuing",
    )
    args = parser.parse_args()

    summary = evaluate_manifest(
        manifest_path=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        threshold_step=args.threshold_step,
        continue_on_error=not args.fail_fast,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
