import json
import os
import tempfile
import unittest

from evaluation.calibration import build_tuned_config, load_summary, select_threshold, write_tuned_config


class EvaluationCalibrationTests(unittest.TestCase):
    def test_select_threshold_prefers_best_f1_row(self):
        summary = {
            "best_f1_threshold": {"threshold": 0.45, "f1": 0.82, "accuracy": 0.8, "balanced_accuracy": 0.79},
            "best_balanced_accuracy_threshold": {"threshold": 0.55, "f1": 0.79, "accuracy": 0.81, "balanced_accuracy": 0.83},
            "threshold_sweep": [
                {"threshold": 0.45, "f1": 0.82, "accuracy": 0.8, "balanced_accuracy": 0.79, "precision": 0.8, "recall": 0.84, "specificity": 0.74},
                {"threshold": 0.55, "f1": 0.79, "accuracy": 0.81, "balanced_accuracy": 0.83, "precision": 0.82, "recall": 0.76, "specificity": 0.9},
            ],
        }
        selected = select_threshold(summary, metric="f1")
        self.assertEqual(selected["threshold"], 0.45)

    def test_select_threshold_supports_other_metrics(self):
        summary = {
            "best_f1_threshold": {"threshold": 0.45, "f1": 0.82, "accuracy": 0.8, "balanced_accuracy": 0.79},
            "best_balanced_accuracy_threshold": {"threshold": 0.55, "f1": 0.79, "accuracy": 0.81, "balanced_accuracy": 0.83},
            "threshold_sweep": [
                {"threshold": 0.45, "f1": 0.82, "accuracy": 0.8, "balanced_accuracy": 0.79, "precision": 0.8, "recall": 0.84, "specificity": 0.74},
                {"threshold": 0.55, "f1": 0.79, "accuracy": 0.81, "balanced_accuracy": 0.83, "precision": 0.82, "recall": 0.76, "specificity": 0.9},
            ],
        }
        selected = select_threshold(summary, metric="specificity")
        self.assertEqual(selected["threshold"], 0.55)

    def test_build_and_write_tuned_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as handle:
                handle.write(
                    "thresholds:\n"
                    "  final_verdict: 0.5\n"
                    "weights:\n"
                    "  syncnet: 0.25\n"
                    "  lipinc: 0.50\n"
                    "  biolip: 0.25\n"
                )

            tuned = build_tuned_config(config_path, 0.42)
            self.assertEqual(tuned["thresholds"]["final_verdict"], 0.42)

            output_path = os.path.join(tmpdir, "tuned.yaml")
            write_tuned_config(tuned, output_path)
            self.assertTrue(os.path.isfile(output_path))

    def test_load_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.json")
            expected = {"best_f1_threshold": {"threshold": 0.4}}
            with open(summary_path, "w") as handle:
                json.dump(expected, handle)
            loaded = load_summary(tmpdir)
            self.assertEqual(loaded, expected)


if __name__ == "__main__":
    unittest.main()
