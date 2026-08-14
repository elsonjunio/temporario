"""Tests for the reusable page map (``snapshot_map``): cached reuse without
re-crawling the DOM, staleness detection by URL and opt-in disk persistence
(``BROWSER_SNAPSHOT_DIR``)."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.browser_fixtures.serve import serve

from src.navigation import controller as controller_module
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
class TestSnapshotMap(unittest.TestCase):
    server: Any
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.controller = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(self.controller.close)

    def _open(self, path: str = "/index.html") -> None:
        result = self.controller.open(f"{self.base_url}{path}")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")

    def test_map_returns_cached_without_recrawl(self):
        self._open()
        with mock.patch.object(
            controller_module, "build_snapshot", wraps=controller_module.build_snapshot
        ) as build:
            snap = self.controller.snapshot()
            self.assertEqual(snap["status"], "success")
            self.assertEqual(build.call_count, 1)

            page_map = self.controller.snapshot_map()
            self.assertEqual(page_map["status"], "success")
            self.assertEqual(page_map["found"], True)
            self.assertEqual(page_map["fresh"], True)
            self.assertEqual(page_map["stale"], False)
            self.assertEqual(page_map["source"], "live")
            self.assertEqual(page_map["url"], f"{self.base_url}/index.html")
            self.assertEqual(page_map["epoch"], snap["epoch"])
            self.assertEqual(page_map["ref_count"], snap["ref_count"])
            self.assertEqual(page_map["content"], snap["content"])
            self.assertEqual(page_map["refs"], snap["refs"])
            self.assertEqual(build.call_count, 1)

    def test_map_is_stale_after_navigation(self):
        self._open("/index.html")
        snap = self.controller.snapshot()
        self._open("/about.html")
        page_map = self.controller.snapshot_map()
        self.assertEqual(page_map["found"], True)
        self.assertEqual(page_map["fresh"], False)
        self.assertEqual(page_map["stale"], True)
        self.assertEqual(page_map["url"], f"{self.base_url}/index.html")
        self.assertEqual(page_map["content"], snap["content"])

    def test_map_refresh_forces_recrawl(self):
        self._open()
        with mock.patch.object(
            controller_module, "build_snapshot", wraps=controller_module.build_snapshot
        ) as build:
            self.controller.snapshot()
            refreshed = self.controller.snapshot_map(refresh=True)
            self.assertEqual(refreshed["status"], "success")
            self.assertEqual(refreshed["operation"], "snapshot")
            self.assertEqual(build.call_count, 2)

    def test_map_not_found(self):
        self._open()
        page_map = self.controller.snapshot_map(url="http://127.0.0.1:9/nope.html")
        self.assertEqual(page_map["status"], "success")
        self.assertEqual(page_map["found"], False)

    def test_map_persists_to_disk_between_sessions(self):
        with tempfile.TemporaryDirectory() as snapshot_dir:
            self.controller.close()
            first = BrowserController(
                action_timeout=20, nav_timeout=20, snapshot_dir=snapshot_dir
            )
            first.open(f"{self.base_url}/index.html")
            first.wait(condition="network_idle")
            snap = first.snapshot()
            self.assertEqual(snap["status"], "success")
            first.close()

            second = BrowserController(
                action_timeout=20, nav_timeout=20, snapshot_dir=snapshot_dir
            )
            self.addCleanup(second.close)
            page_map = second.snapshot_map(url=f"{self.base_url}/index.html")
            self.assertEqual(page_map["status"], "success")
            self.assertEqual(page_map["found"], True)
            self.assertEqual(page_map["source"], "disk")
            self.assertFalse(second.session.is_open)
            self.assertEqual(page_map["content"], snap["content"])
            self.assertEqual(page_map["ref_count"], snap["ref_count"])

    def test_map_file_naming_is_deterministic(self):
        from src.navigation.snapshot import snapshot_map_file

        with tempfile.TemporaryDirectory() as snapshot_dir:
            url = f"{self.base_url}/index.html"
            first = snapshot_map_file(snapshot_dir, url)
            second = snapshot_map_file(snapshot_dir, url)
            self.assertEqual(first, second)
            self.assertEqual(first.suffix, ".json")
            self.assertTrue(first.name.startswith("map-"))
            other = snapshot_map_file(snapshot_dir, f"{self.base_url}/about.html")
            self.assertNotEqual(first, other)
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            first.write_text("not json", encoding="utf-8")
            from src.navigation.snapshot import read_map_file

            self.assertIsNone(read_map_file(snapshot_dir, url))


if __name__ == "__main__":
    unittest.main()
