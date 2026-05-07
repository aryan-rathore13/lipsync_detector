import os
import tempfile
import unittest

from evaluation import harness


class EvaluationHarnessTests(unittest.TestCase):
    def test_compute_metrics(self):
        metrics = harness.compute_metrics(
            labels=[1, 1, 0, 0],
            scores=[0.9, 0.6, 0.7, 0.1],
            threshold=0.5,
        )

        self.assertEqual(metrics["confusion_matrix"], {"tp": 2, "tn": 1, "fp": 1, "fn": 0})
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.8)

    def test_compute_roc_auc(self):
        auc = harness.compute_roc_auc(
            labels=[1, 1, 0, 0],
            scores=[0.9, 0.8, 0.4, 0.2],
        )
        self.assertEqual(auc, 1.0)

    def test_evaluate_manifest_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.csv")
            config_path = os.path.join(tmpdir, "config.yaml")
            output_dir = os.path.join(tmpdir, "results")

            with open(manifest_path, "w") as handle:
                handle.write(
                    "sample_id,video_path,label,split,language,source\n"
                    "real_1,/tmp/real.mp4,REAL,test,english,synthetic\n"
                    "fake_1,/tmp/fake.mp4,FAKE,test,hindi,synthetic\n"
                )

            with open(config_path, "w") as handle:
                handle.write(
                    "thresholds:\n"
                    "  final_verdict: 0.5\n"
                    "weights:\n"
                    "  syncnet: 0.25\n"
                    "  lipinc: 0.50\n"
                    "  biolip: 0.25\n"
                )

            fake_results = {
                "/tmp/real.mp4": {"verdict": "REAL", "confidence": 0.2, "triggered_by": "ensemble", "processing_time_s": 1.2},
                "/tmp/fake.mp4": {"verdict": "FAKE", "confidence": 0.9, "triggered_by": "boolean_gate", "processing_time_s": 1.0},
            }

            def fake_detect(video_path, cfg, verbose):
                self.assertFalse(verbose)
                return fake_results[video_path]

            summary = harness.evaluate_manifest(
                manifest_path=manifest_path,
                config_path=config_path,
                output_dir=output_dir,
                threshold_step=0.5,
                detect_fn=fake_detect,
            )

            self.assertEqual(summary["num_samples"], 2)
            self.assertEqual(summary["num_successful"], 2)
            self.assertEqual(summary["num_failed"], 0)
            self.assertEqual(summary["metrics_at_operating_threshold"]["accuracy"], 1.0)
            self.assertEqual(summary["roc_auc"], 1.0)

            run_dir = summary["run_dir"]
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "predictions.csv")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "predictions.jsonl")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "threshold_sweep.csv")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "report.md")))
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "plots")))


if __name__ == "__main__":
    unittest.main()
