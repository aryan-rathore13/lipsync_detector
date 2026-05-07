#!/usr/bin/env bash

set -euo pipefail

mkdir -p models/syncnet models/lipinc_v2 models/biolip

download_if_missing() {
  local url="$1"
  local output="$2"

  if [ -f "$output" ]; then
    echo "[skip] $output already exists"
    return 0
  fi

  echo "[download] $output"
  curl -L "$url" -o "$output"
}

download_if_missing \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  "models/face_landmarker.task"

download_if_missing \
  "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model" \
  "models/syncnet/syncnet_v2.model"

echo
echo "LipForensics weights are not auto-downloaded here because the original release is distributed via Google Drive."
echo "Place the checkpoint at: models/lipinc_v2/lipforensics_ff.pth"
