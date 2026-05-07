#!/usr/bin/env python3

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown and plots for an evaluation run directory.")
    parser.add_argument("run_dir", help="Path to an evaluation run directory containing summary.json and predictions.csv")
    args = parser.parse_args()

    artifacts = generate_report(args.run_dir)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
