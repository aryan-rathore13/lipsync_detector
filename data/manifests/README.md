# Manifest Guide

Use CSV manifests to define reproducible evaluation slices.

Required columns:

- `video_path`
- `label`

Recommended columns:

- `sample_id`
- `split`
- `language`
- `source`
- `notes`

Example workflow:

1. Start from a template in `data/manifests/templates/`.
2. Fill in absolute paths or paths relative to the manifest file.
3. Keep dataset slices separate, for example `val_celebdf.csv`, `test_hindi_custom.csv`, or `wav2lip_ablation.csv`.
4. Run `python scripts/evaluate.py <manifest.csv>`.

Suggested split discipline:

- `train`: only if you later add threshold fitting or classifier training
- `val`: threshold tuning and ablation comparison
- `test`: final held-out reporting only
