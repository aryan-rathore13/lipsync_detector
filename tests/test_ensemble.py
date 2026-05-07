import unittest

from pipeline import ensemble


class EnsembleTests(unittest.TestCase):
    def test_boolean_gate_short_circuits(self):
        cfg = {
            "weights": {"syncnet": 0.25, "lipinc": 0.5, "biolip": 0.25},
            "thresholds": {"final_verdict": 0.5, "syncnet_veto": 0.5},
        }

        result = ensemble.fuse(
            {"triggered": True, "score": 0.9},
            {},
            {},
            {},
            cfg,
        )

        self.assertEqual(result["verdict"], "FAKE")
        self.assertEqual(result["triggered_by"], "boolean_gate")
        self.assertEqual(result["confidence"], 1.0)

    def test_weighted_average_skips_missing_layers(self):
        cfg = {
            "weights": {"syncnet": 0.25, "lipinc": 0.5, "biolip": 0.25},
            "thresholds": {"final_verdict": 0.5, "syncnet_veto": 0.5},
        }

        result = ensemble.fuse(
            {"triggered": False, "mismatch_ratio": 0.1},
            {"skipped": True, "lse_c": None, "offset_frames": 0},
            {"lipinc_fake_score": 0.8, "method": "lipforensics"},
            {"biolip_fake_score": 0.2, "method": "kinematic_fallback"},
            cfg,
        )

        self.assertEqual(result["triggered_by"], "ensemble")
        self.assertEqual(result["verdict"], "FAKE")
        self.assertAlmostEqual(result["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
