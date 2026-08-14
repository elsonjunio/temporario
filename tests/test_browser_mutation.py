"""Tests for automatic ref invalidation via the in-page MutationObserver:
structural DOM changes (or navigation) invalidate snapshot refs, a fresh
snapshot clears the state, and BROWSER_MUTATION_INVALIDATE=0 disables the
blocking (the mutated flag stays observable)."""

from __future__ import annotations

import threading
import unittest
from typing import Any

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
class TestMutationInvalidation(unittest.TestCase):
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

    def _refs(self) -> dict[str, str]:
        snap = self.controller.snapshot()
        self.assertEqual(snap["status"], "success")
        return {meta["name"]: ref for ref, meta in snap["refs"].items()}

    def _mutate_structurally(self) -> None:
        result = self.controller.evaluate(
            "return (function(){ var d = document.createElement('div'); "
            "d.textContent = 'extra'; document.body.appendChild(d); return true; })()"
        )
        self.assertEqual(result["status"], "success")

    def test_structural_mutation_invalidates_refs(self):
        self._open()
        refs = self._refs()
        self._mutate_structurally()

        page_map = self.controller.snapshot_map()
        self.assertEqual(page_map["found"], True)
        self.assertEqual(page_map["mutated"], True)
        self.assertEqual(page_map["fresh"], False)
        self.assertEqual(page_map["stale"], True)

        clicked = self.controller.click(ref=refs["Disparar erro JS"])
        self.assertEqual(clicked["status"], "error")
        self.assertEqual(clicked["error"]["type"], "invalid_reference")
        self.assertIn("stale", clicked["error"]["message"].lower())

    def test_fresh_snapshot_clears_mutation_state(self):
        self._open()
        refs = self._refs()
        self._mutate_structurally()
        self._refs()

        self.assertEqual(self.controller.status()["mutated"], False)
        clicked = self.controller.click(ref=refs["Disparar erro JS"])
        self.assertEqual(clicked["status"], "success")

    def test_non_structural_change_keeps_refs(self):
        self._open()
        refs = self._refs()
        result = self.controller.evaluate(
            "return (function(){ var h = document.querySelector('h1'); "
            "if (h) h.setAttribute('data-x', '1'); return true; })()"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.controller.status()["mutated"], False)
        clicked = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(clicked["status"], "success")

    def test_fill_does_not_invalidate_refs(self):
        self._open()
        refs = self._refs()
        filled = self.controller.fill(ref=refs["Nome"], value="Maria Souza")
        self.assertEqual(filled["status"], "success")
        self.assertEqual(self.controller.status()["mutated"], False)
        clicked = self.controller.click(ref=refs["Criar usuário"])
        self.assertEqual(clicked["status"], "success")

    def test_structural_dom_edit_tool_invalidates_refs(self):
        self._open()
        refs = self._refs()
        inserted = self.controller.insert_html(html="<div>extra</div>", selector="body")
        self.assertEqual(inserted["status"], "success")
        clicked = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(clicked["status"], "error")
        self.assertEqual(clicked["error"]["type"], "invalid_reference")

    def test_url_change_invalidates_refs(self):
        self._open("/index.html")
        refs = self._refs()
        # SPA-style URL change (pushState): no document reload, no DOM mutation,
        # so only the URL guard can invalidate the refs.
        pushed = self.controller.evaluate(
            "return (function(){ history.pushState({}, '', '/spa'); "
            "return location.href; })()"
        )
        self.assertEqual(pushed["status"], "success")
        self.assertEqual(self.controller.status()["mutated"], False)
        clicked = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(clicked["status"], "error")
        self.assertEqual(clicked["error"]["type"], "invalid_reference")

    def test_full_navigation_blocks_refs(self):
        self._open("/index.html")
        refs = self._refs()
        self._open("/about.html")
        clicked = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(clicked["status"], "error")
        self.assertEqual(clicked["error"]["type"], "invalid_reference")

    def test_disabled_via_mutation_invalidate_false(self):
        self.controller.close()
        self.controller = BrowserController(
            action_timeout=20, nav_timeout=20, mutation_invalidate=False
        )
        self.addCleanup(self.controller.close)
        self._open()
        refs = self._refs()
        self._mutate_structurally()
        self.assertEqual(self.controller.status()["mutated"], True)
        clicked = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(clicked["status"], "success")

    def test_status_reports_mutated(self):
        self._open()
        self._refs()
        self.assertEqual(self.controller.status()["mutated"], False)
        self._mutate_structurally()
        self.assertEqual(self.controller.status()["mutated"], True)
        self._refs()
        self.assertEqual(self.controller.status()["mutated"], False)


if __name__ == "__main__":
    unittest.main()
