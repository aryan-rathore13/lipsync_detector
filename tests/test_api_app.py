import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.app import create_app


class ApiAppTests(unittest.TestCase):
    def test_health_endpoint(self):
        app = create_app(config_loader=lambda path: {"thresholds": {"final_verdict": 0.5}})
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("config_path", body)

    def test_detect_path_success(self):
        called = {}

        def fake_detect(video_path, cfg, verbose):
            called["video_path"] = video_path
            called["cfg"] = cfg
            called["verbose"] = verbose
            return {"verdict": "REAL", "confidence": 0.12, "triggered_by": "ensemble"}

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            app = create_app(
                detect_fn=fake_detect,
                config_loader=lambda path: {"thresholds": {"final_verdict": 0.5}},
            )
            client = TestClient(app)
            response = client.post("/detect/path", json={"video_path": handle.name})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"]["mode"], "path")
        self.assertEqual(body["result"]["verdict"], "REAL")
        self.assertFalse(called["verbose"])
        self.assertEqual(called["video_path"], os.path.abspath(handle.name))

    def test_detect_path_missing_file(self):
        app = create_app(config_loader=lambda path: {"thresholds": {"final_verdict": 0.5}})
        client = TestClient(app)

        response = client.post("/detect/path", json={"video_path": "/tmp/does_not_exist.mp4"})

        self.assertEqual(response.status_code, 404)

    def test_detect_upload_success(self):
        seen = {}

        def fake_detect(video_path, cfg, verbose):
            seen["exists_during_detect"] = os.path.exists(video_path)
            seen["video_path"] = video_path
            return {"verdict": "FAKE", "confidence": 0.91, "triggered_by": "boolean_gate"}

        app = create_app(
            detect_fn=fake_detect,
            config_loader=lambda path: {"thresholds": {"final_verdict": 0.5}},
        )
        client = TestClient(app)

        response = client.post(
            "/detect/upload",
            files={"file": ("clip.mp4", b"fake-binary-video", "video/mp4")},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"]["mode"], "upload")
        self.assertEqual(body["source"]["filename"], "clip.mp4")
        self.assertTrue(seen["exists_during_detect"])
        self.assertFalse(os.path.exists(seen["video_path"]))

    def test_detect_upload_rejects_extension(self):
        app = create_app(config_loader=lambda path: {"thresholds": {"final_verdict": 0.5}})
        client = TestClient(app)

        response = client.post(
            "/detect/upload",
            files={"file": ("clip.txt", b"not-a-video", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
