#!/usr/bin/env python3

import argparse
import os
import sys

import uvicorn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lip-sync detector HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
