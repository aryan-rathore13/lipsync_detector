"""
Main entry point.

Usage:
    python detector.py path/to/video.mp4
    python detector.py path/to/video.mp4 --config config.yaml

Returns JSON:
{
    "verdict": "FAKE" | "REAL" | "UNKNOWN",
    "confidence": 0.0–1.0,
    "triggered_by": "boolean_gate" | "syncnet_veto" | "ensemble",
    "layer_scores": { ... },
    "syncnet_offset_frames": int,
    "processing_time_s": float
}
"""

import argparse
import contextlib
import io
import json
import os
import time

import yaml

from pipeline import preprocessor, boolean_gate, syncnet_scorer, lipinc_scorer, biolip_scorer, ensemble


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_detection_pipeline(video_path: str, cfg: dict, verbose: bool) -> dict:
    t0 = time.time()

    if verbose:
        print(f"[1/5] Preprocessing: {video_path}")
    data = preprocessor.preprocess(video_path, cfg)
    if verbose:
        print(f"      → {len(data['lip_frames'])} frames, {len(data['audio_wav'])/data['sample_rate']:.1f}s audio")

    if verbose:
        print("[2/5] Boolean gate check")
    gate = boolean_gate.run(data, cfg)
    if verbose:
        print(f"      → mismatch_ratio={gate['mismatch_ratio']:.3f}  triggered={gate['triggered']}")

    if gate["triggered"]:
        result = ensemble.fuse(gate, {}, {}, {}, cfg)
        result["processing_time_s"] = round(time.time() - t0, 2)
        return result

    if verbose:
        print("[3/5] SyncNet temporal sync scoring")
    sync = syncnet_scorer.run(data, cfg, video_path=video_path)
    if verbose:
        if sync.get("skipped"):
            print(f"      → SKIPPED: {sync.get('reason')}")
        else:
            print(f"      → LSE-C={sync['lse_c']:.2f}  LSE-D={sync['lse_d']:.2f}  "
                  f"offset={sync['offset_frames']} frames  "
                  f"fake_score={sync['syncnet_fake_score']:.3f}")

    if verbose:
        print("[4/5] LIPINC-V2 / LipForensics spatiotemporal scoring")
    lipinc = lipinc_scorer.run(data, cfg)
    if verbose:
        print(f"      → method={lipinc['method']}  fake_score={lipinc['lipinc_fake_score']:.3f}")

    if verbose:
        print("[5/5] BioLip biomechanical scoring")
    biolip = biolip_scorer.run(data, cfg)
    if verbose:
        print(f"      → method={biolip['method']}  fake_score={biolip['biolip_fake_score']:.3f}")

    result = ensemble.fuse(gate, sync, lipinc, biolip, cfg)
    result["processing_time_s"] = round(time.time() - t0, 2)
    return result


def detect(video_path: str, cfg: dict, verbose: bool = True) -> dict:
    if verbose:
        return _run_detection_pipeline(video_path, cfg, verbose=True)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _run_detection_pipeline(video_path, cfg, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Lip-sync deepfake detector")
    parser.add_argument("video", help="Path to input video file (MP4/AVI)")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = detect(args.video, cfg)

    print("\n" + "=" * 50)
    print(f"  VERDICT    : {result['verdict']}")
    print(f"  CONFIDENCE : {result['confidence']:.3f}")
    print(f"  TIME       : {result['processing_time_s']}s")
    print("=" * 50)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
