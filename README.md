# Lip-Sync Deepfake Detector

An audio-visual deepfake detection project focused on lip-sync integrity rather than speaker identity. The repository combines a lightweight rule-based gate, a temporal sync model, a lip-region forgery scorer, and a biomechanical fallback into a single CLI pipeline.

This is not a foundation model or a fully original detector from scratch. The core contribution in this repo is the integration layer: preprocessing, model orchestration, fallback heuristics, score fusion, thresholding, and project packaging around established research components.

## What This Repo Implements

- A single-command detector entrypoint in [detector.py](detector.py)
- Lip-region extraction and mouth motion features using MediaPipe
- A fast boolean gate for obvious audio/lip mismatches
- SyncNet-based temporal alignment scoring
- LipForensics-based spatiotemporal scoring when weights are available
- A kinematic fallback that scores implausible lip motion when a dedicated biomechanical model is not available
- Score fusion with configurable thresholds and weights in [config.yaml](config.yaml)

## Current Status

What works today:

- The English / Indian-English path has been smoke-tested locally.
- The detector can flag obvious frozen-lip, muted-lip, and shifted-audio cases.
- The project supports graceful fallbacks when some optional models are missing.

What is still incomplete:

- Multilingual robustness has not been benchmarked systematically.
- The biomechanical layer is currently a heuristic fallback, not a fully integrated external BioLip model.
- A formal evaluation harness now exists, but it still needs real benchmark manifests and published result tables.
- Multi-face handling is still coarse.
- The MediaPipe Tasks face-landmarker path can be unstable in some headless macOS environments, so local desktop execution is the main supported path today.

That distinction matters if you want to use this repo professionally. A strong portfolio project is better served by clear claims plus measurable next steps than by over-claiming “production grade” without evidence.

## Architecture

```text
Input video
  -> Preprocessing
     - extract mono audio
     - resample video to target FPS
     - track lip region and MAR sequence
  -> Boolean gate
     - audio active but lips static
     - lips moving but audio silent
  -> SyncNet scorer
     - temporal alignment confidence
     - frame offset estimate
  -> LipForensics scorer
     - spatiotemporal lip-region forgery score
     - falls back if weights are absent
  -> Kinematic scorer
     - landmark velocity / acceleration heuristic
  -> Ensemble fusion
     - final fake probability and verdict
```

## Repository Layout

```text
lipsync_detector/
├── detector.py
├── config.yaml
├── pipeline/
├── scripts/
├── tests/
├── models/
├── THIRD_PARTY_NOTICES.md
└── .github/workflows/unit-tests.yml
```

Third-party model repos are intentionally not tracked by default. Fetch them locally with the setup script.

## Setup

### 1. System dependency

```bash
brew install ffmpeg
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Fetch third-party repos

```bash
bash scripts/setup_third_party.sh
```

### 4. Download required model assets

```bash
bash scripts/download_models.sh
```

LipForensics weights are still a manual step. Place the checkpoint at:

```text
models/lipinc_v2/lipforensics_ff.pth
```

## Usage

```bash
source .venv/bin/activate
python detector.py path/to/video.mp4
```

For system integration, you can also run the detector as an HTTP service:

```bash
source .venv/bin/activate
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /detect/path`
- `POST /detect/upload`

Example path request:

```bash
curl -X POST http://127.0.0.1:8000/detect/path \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/absolute/path/to/video.mp4"}'
```

Example upload request:

```bash
curl -X POST http://127.0.0.1:8000/detect/upload \
  -F "file=@/absolute/path/to/video.mp4"
```

Example output:

```json
{
  "verdict": "FAKE",
  "confidence": 0.84,
  "triggered_by": "syncnet_veto",
  "layer_scores": {
    "boolean_gate": 0.32,
    "syncnet": 1.0,
    "lipinc": 0.0,
    "biolip": 0.0
  },
  "syncnet_lse_c": 0.35,
  "syncnet_lse_d": 11.77,
  "syncnet_offset_frames": 6,
  "lipinc_method": "jitter_fallback",
  "biolip_method": "kinematic_fallback",
  "processing_time_s": 27.4
}
```

## Configuration

All thresholds and layer weights live in [config.yaml](config.yaml).

Key knobs:

- `thresholds.boolean_gate_mismatch_ratio`: how often audio/lip disagreement must occur before an instant fake verdict
- `thresholds.syncnet_veto`: hard lower bound for severe desynchronization
- `weights.syncnet`: contribution from temporal alignment
- `weights.lipinc`: contribution from LipForensics or its fallback
- `weights.biolip`: contribution from the biomechanical heuristic

## Tests

Lightweight unit tests are included for the boolean gate and ensemble logic.

```bash
make test
```

GitHub Actions is configured to run these tests automatically on push and pull request.

## Evaluation Harness

The repo now includes a manifest-driven evaluation harness in [scripts/evaluate.py](scripts/evaluate.py).

Manifest format:

```csv
sample_id,video_path,label,split,language,source,notes
clip_001,/abs/path/video1.mp4,REAL,test,english,celebdf,clean sample
clip_002,/abs/path/video2.mp4,FAKE,test,hindi,custom,wav2lip variant
```

Minimum required columns:

- `video_path`
- `label`

Run it with:

```bash
python scripts/evaluate.py data/manifests/sample_eval_manifest.csv
```

Artifacts are written under `results/eval/<timestamp>/`:

- `predictions.jsonl`: one row per sample with raw detector outputs
- `predictions.csv`: spreadsheet-friendly export
- `threshold_sweep.csv`: metrics across thresholds from 0.0 to 1.0
- `summary.json`: aggregate metrics, ROC-AUC, best F1 threshold, grouped summaries
- `config_snapshot.json`: exact config used for the run
- `report.md`: markdown report you can attach to the repo or a benchmark folder
- `plots/`: threshold sweep, confidence distribution, and decision breakdown charts

Reusable templates are included in [data/manifests/templates](data/manifests/templates), and manifest conventions are documented in [data/manifests/README.md](data/manifests/README.md).

This is the right foundation for adding held-out validation splits, multilingual slices, and later threshold calibration.

Recommended workflow:

1. Build a validation manifest, for example `data/manifests/val_hindi.csv`.
2. Run `python scripts/evaluate.py data/manifests/val_hindi.csv`.
3. Use the generated run directory with `python scripts/calibrate_threshold.py results/eval/<run_id> --metric f1 --write-config config.val_tuned.yaml`.
4. Evaluate the tuned config on a separate held-out test manifest.

This keeps threshold selection honest: tune on `val`, report on `test`.

## Suggested Next Improvements

If your goal is a stronger AI/ML portfolio project, these are the highest-value upgrades:

1. Populate the evaluation harness with real benchmark manifests.
Use the current harness against real held-out slices, multilingual subsets, and architecture ablations. The credibility gain now comes from the data and reporting discipline, not from more harness code.

2. Add multilingual validation.
Test on Hindi, Bengali, Tamil, and mixed-code speech. Right now the repo can claim “not yet fully validated outside English,” not “language agnostic.”

3. Replace heuristic thresholds with calibrated ones.
Persist experiment results and tune thresholds on a held-out validation split instead of hand-tuning from a few samples.

4. Support per-face analysis in multi-person scenes.
Track each face separately and return per-face sync scores. This makes the project much more deployment-relevant.

5. Add an API layer before a frontend.
Expose the detector through FastAPI with async job submission, file upload validation, and structured JSON responses. That is more useful than building a UI first.

6. Add experiment tracking.
Save run metadata, config snapshots, and outputs to a `results/` directory or MLflow/W&B. Hiring managers notice reproducibility.

7. Package sample artifacts properly.
Do not commit large raw videos or heavyweight checkpoints directly into git. Use GitHub Releases, Hugging Face, or an object store for assets.

## Should You Build a Frontend?

Yes, but only after an API and evaluation layer exist.

The best sequence is:

1. Stabilize the detector and benchmark it.
2. Wrap it with FastAPI.
3. Add a minimal frontend for upload, progress, verdict, and per-layer diagnostics.

A frontend helps for demos, but it does not compensate for weak validation. For hiring, a clean API plus evaluation report is worth more than a flashy UI on top of uncertain metrics.

## Third-Party Components

This project depends on external research code and pretrained checkpoints. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing notes.

If you later decide to vendor upstream code directly into this repository, keep the original license files and make that provenance explicit.
