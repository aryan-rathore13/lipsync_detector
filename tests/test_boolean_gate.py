import unittest

import numpy as np

from pipeline import boolean_gate


class BooleanGateTests(unittest.TestCase):
    def test_triggers_when_audio_and_lips_mismatch(self):
        data = {
            "audio_wav": np.ones(32000, dtype=np.float32),
            "mar_sequence": np.zeros(50, dtype=np.float32),
            "fps": 25.0,
            "sample_rate": 16000,
        }
        cfg = {
            "thresholds": {
                "rms_energy": 0.001,
                "mar_variance": 0.0001,
                "boolean_gate_mismatch_ratio": 0.5,
            }
        }

        result = boolean_gate.run(data, cfg)

        self.assertTrue(result["triggered"])
        self.assertAlmostEqual(result["mismatch_ratio"], 1.0)

    def test_returns_inconclusive_for_empty_windows(self):
        data = {
            "audio_wav": np.array([], dtype=np.float32),
            "mar_sequence": np.array([], dtype=np.float32),
            "fps": 25.0,
            "sample_rate": 16000,
        }
        cfg = {
            "thresholds": {
                "rms_energy": 0.001,
                "mar_variance": 0.0001,
                "boolean_gate_mismatch_ratio": 0.5,
            }
        }

        result = boolean_gate.run(data, cfg)

        self.assertFalse(result["triggered"])
        self.assertEqual(result["mismatch_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
