#!/usr/bin/env bash

set -euo pipefail

clone_if_missing() {
  local repo_url="$1"
  local dest_dir="$2"

  if [ -d "$dest_dir/.git" ] || [ -d "$dest_dir" ]; then
    echo "[skip] $dest_dir already exists"
    return 0
  fi

  echo "[clone] $repo_url -> $dest_dir"
  git clone "$repo_url" "$dest_dir"
}

clone_if_missing "https://github.com/joonson/syncnet_python.git" "syncnet_python"
clone_if_missing "https://github.com/ahaliassos/LipForensics.git" "lipforensics"

echo
echo "Third-party repos are present."
echo "Next: bash scripts/download_models.sh"
