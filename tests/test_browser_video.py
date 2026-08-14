"""Tests for the media diagnostics: video recording (opt-in via
``video_dir`` / ``BROWSER_VIDEO_DIR``), the ``video``/``video_save`` actions
and the ``inspect(include=['trace'])`` trace+screenshot capture."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from typing import Any
from unittest import mock

from tests.browser_fixtures.serve import serve

from src.navigation.controller import BrowserController

PLAYWRIGHT_AVAILABLE = False
try:  # pragma: no cover - environment check
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    _probe = _pw.chromium.launch(headless=True)
    _probe.close()
    _pw.stop()
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


def _run_server():
    server = serve()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestVideoRecording(unittest.TestCase):
    server: Any
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.video_dir = os.path.join(self.tmp, "videos")
        self.controller = BrowserController(
            action_timeout=20, nav_timeout=20, video_dir=self.video_dir
        )
        self.addCleanup(self.controller.close)

    def _open(self, path: str = "/index.html") -> None:
        result = self.controller.open(f"{self.base_url}{path}")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")

    def test_video_disabled_reports_cleanly(self):
        other = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(other.close)
        info = other.video()
        self.assertEqual(info["status"], "success")
        self.assertEqual(info["recording"], False)
        self.assertEqual(info["finalized"], [])
        self.assertEqual(other.status()["video"]["recording"], False)
        other.open(f"{self.base_url}/index.html")
        saved = other.video_save()
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["error"]["type"], "browser_error")
        self.assertTrue(saved["error"]["recoverable"])

    def test_video_reports_recording_state(self):
        self._open()
        info = self.controller.video()
        self.assertEqual(info["recording"], True)
        self.assertEqual(info["finalized"], [])
        self.assertTrue(info["path"])
        self.assertEqual(self.controller.status()["video"]["recording"], True)

    def test_video_save_while_recording_is_recoverable_error(self):
        self._open()
        result = self.controller.video_save()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "browser_error")
        self.assertIn("recording in progress", result["error"]["message"])
        self.assertTrue(result["error"]["recoverable"])

    def test_video_save_saves_closed_page_recording(self):
        self._open()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        self.controller.close_tab(index=0)
        result = self.controller.video_save()
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["saved"]), 1)
        entry = result["saved"][0]
        self.assertTrue(os.path.exists(entry["path"]))
        self.assertGreater(entry["bytes"], 0)
        self.assertTrue(entry["path"].endswith(".webm"))
        self.assertEqual(len(result["pending"]), 1)
        again = self.controller.video_save()
        self.assertEqual(again["status"], "success")
        self.assertEqual(len(again["saved"]), 0)

    def test_close_finalizes_video_into_video_dir(self):
        self._open()
        target = self.controller.video()["path"]
        self.controller.close()
        self.assertTrue(os.path.exists(target))
        self.assertGreater(os.path.getsize(target), 0)


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestInspectTraceAndVideo(unittest.TestCase):
    server: Any
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.controller = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(self.controller.close)

    def _open(self) -> None:
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")
        self.controller.snapshot()

    def test_inspect_trace_captures_and_extracts_screenshots(self):
        with mock.patch.dict(
            os.environ, {"BROWSER_TRACE_DIR": os.path.join(self.tmp, "traces")}
        ):
            controller = BrowserController(action_timeout=20, nav_timeout=20)
            self.addCleanup(controller.close)
            result = controller.open(f"{self.base_url}/index.html")
            self.assertEqual(result["status"], "success")
            controller.wait(condition="network_idle")
            trace = controller.inspect(include=["trace"])["trace"]
        self.assertTrue(trace["path"])
        self.assertTrue(os.path.exists(trace["path"]))
        self.assertGreaterEqual(trace["frames"], 1)
        self.assertEqual(len(trace["screenshots"]), trace["frames"])
        for shot in trace["screenshots"]:
            self.assertTrue(os.path.exists(shot))
            self.assertGreater(os.path.getsize(shot), 0)

    def test_inspect_video_include(self):
        self._open()
        video = self.controller.inspect(include=["video"])["video"]
        self.assertEqual(video["recording"], False)
        self.assertEqual(video["path"], None)


if __name__ == "__main__":
    unittest.main()
