import json
import os
import tempfile
import unittest

from evaluation.reporting import generate_report


class EvaluationReportingTests(unittest.TestCase):
    def test_generate_report_creates_markdown_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {
                "manifest_path": "/tmp/manifest.csv",
                "config_path": "/tmp/config.yaml",
                "num_samples": 2,
                "num_successful": 2,
                "num_failed": 0,
                "operating_threshold": 0.5,
                "mean_processing_time_s": 1.1,
                "roc_auc": 1.0,
                "metrics_at_operating_threshold": {
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "specificity": 1.0,
                    "balanced_accuracy": 1.0,
                    "confusion_matrix": {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
                },
                "best_f1_threshold": {
                    "threshold": 0.5,
                    "accuracy": 1.0,
                    "f1": 1.0,
                    "balanced_accuracy": 1.0,
                },
                "best_balanced_accuracy_threshold": {
                    "threshold": 0.5,
                    "accuracy": 1.0,
                    "f1": 1.0,
                    "balanced_accuracy": 1.0,
                },
                "triggered_by_breakdown": {"ensemble": 1, "boolean_gate": 1},
                "group_summaries": {
                    "language": {
                        "english": {
                            "support": 2,
                            "accuracy": 1.0,
                            "f1": 1.0,
                            "balanced_accuracy": 1.0,
                            "roc_auc": 1.0,
                        }
                    }
                },
                "failures": [],
            }
            with open(os.path.join(tmpdir, "summary.json"), "w") as handle:
                json.dump(summary, handle)

            with open(os.path.join(tmpdir, "predictions.csv"), "w") as handle:
                handle.write(
                    "sample_id,status,label,confidence,triggered_by\n"
                    "real_1,ok,0,0.1,ensemble\n"
                    "fake_1,ok,1,0.9,boolean_gate\n"
                )

            with open(os.path.join(tmpdir, "threshold_sweep.csv"), "w") as handle:
                handle.write(
                    "threshold,accuracy,f1,balanced_accuracy\n"
                    "0.0,0.5,0.667,0.5\n"
                    "0.5,1.0,1.0,1.0\n"
                    "1.0,0.5,0.0,0.5\n"
                )

            artifacts = generate_report(tmpdir)

            self.assertTrue(os.path.isfile(artifacts["report_path"]))
            self.assertTrue(os.path.isfile(artifacts["plot_paths"]["threshold_sweep"]))
            self.assertTrue(os.path.isfile(artifacts["plot_paths"]["score_distribution"]))
            self.assertTrue(os.path.isfile(artifacts["plot_paths"]["trigger_breakdown"]))

            with open(artifacts["report_path"]) as handle:
                report_text = handle.read()
            self.assertIn("# Evaluation Report", report_text)
            self.assertIn("## Metrics At Operating Threshold", report_text)
            self.assertIn("### By language", report_text)
