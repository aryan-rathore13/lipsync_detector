# Third-Party Notices

This repository contains original orchestration code for a lip-sync deepfake detector, plus adapters around external open-source components and pretrained weights.

## Upstream components used by this project

1. SyncNet
Repository: https://github.com/joonson/syncnet_python
License: MIT
Usage in this project: temporal audio-video synchronization scoring

2. LipForensics
Repository: https://github.com/ahaliassos/LipForensics
License: MIT
Usage in this project: spatiotemporal lip-region forgery scoring

3. MediaPipe Face Landmarker
Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
Usage in this project: lip-region localization and simple motion features

## Notes

- The code in this repository does not claim ownership of upstream models, papers, or pretrained checkpoints.
- If you vendor third-party code directly into this repository later, keep the original license files with the vendored source.
- If you redistribute pretrained weights, verify the license and redistribution terms for each checkpoint first.
