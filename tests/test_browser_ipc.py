"""End-to-end tests for the subprocess-isolated browser worker.

Drives ``BrowserClient`` (and the ``create_navigation_tool(isolated=True)``
tool) over a JSON-lines pipe against ``src.navigation.worker``. Requires the
same Playwright + Chromium as the in-process suite.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from tests.browser_fixtures.serve import serve

from src.navigation.ipc import BrowserClient
from src.navigation import create_navigation_tool

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


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestBrowserIPC(unittest.TestCase):
    server: Any
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.client = BrowserClient()
        self.addCleanup(self.client.close)

    def test_open_snapshot_evaluate(self):
        opened = self.client.open(url=f"{self.base_url}/index.html")
        self.assertEqual(opened["status"], "success")
        self.client.wait(condition="network_idle")
        snap = self.client.snapshot()
        self.assertEqual(snap["status"], "success")
        self.assertGreater(snap["ref_count"], 0)
        title = self.client.evaluate(expression="return document.title")
        self.assertEqual(title["result"], "App de Teste")

    def test_multitab_over_pipe(self):
        self.client.open(url=f"{self.base_url}/index.html")
        self.client.new_tab(url=f"{self.base_url}/about.html")
        status = self.client.status()
        self.assertEqual(status["tabs"], 2)
        self.assertEqual(status["active_index"], 1)
        switched = self.client.switch_tab(url="/about.html")
        self.assertEqual(switched["active_index"], 1)
        switched = self.client.switch_tab(index=0)
        self.assertEqual(switched["active_index"], 0)
        listing = self.client.list_tabs()
        self.assertEqual(listing["count"], 2)
        self.client.close_tab(index=1)
        self.assertEqual(self.client.status()["tabs"], 1)

    def test_structured_errors_cross_pipe(self):
        self.client.open(url=f"{self.base_url}/index.html")
        result = self.client.click(ref="e999")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_reference")
        self.assertTrue(self.client.is_open)
        ok = self.client.snapshot()
        self.assertEqual(ok["status"], "success")

    def test_invalid_url_rejected_in_worker(self):
        result = self.client.open(url="file:///etc/passwd")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "navigation_error")
        self.assertTrue(self.client.is_open)

    def test_mutation_invalidation_over_pipe(self):
        tool = create_navigation_tool(FakeProvider(["ok"]), isolated=True, max_steps=4)
        self.addCleanup(tool.dispatch, "close")
        self.assertEqual(
            tool.dispatch("open", url=f"{self.base_url}/index.html")["status"],
            "success",
        )
        snap = tool.dispatch("snapshot")
        self.assertEqual(snap["status"], "success")
        name_to_ref = {meta["name"]: ref for ref, meta in snap["refs"].items()}
        mutated = tool.dispatch(
            "evaluate",
            expression=(
                "return (function(){ var d = document.createElement('div'); "
                "d.textContent = 'extra'; document.body.appendChild(d); "
                "return true; })()"
            ),
        )
        self.assertEqual(mutated["status"], "success")
        status = tool.dispatch("status")
        self.assertEqual(status["mutated"], True)
        clicked = tool.dispatch("click", ref=name_to_ref["Mostrar painel"])
        self.assertEqual(clicked["status"], "error")
        self.assertEqual(clicked["error"]["type"], "invalid_reference")
        fresh = tool.dispatch("snapshot")
        self.assertEqual(fresh["status"], "success")
        self.assertEqual(tool.dispatch("status")["mutated"], False)

    def test_snapshot_map_over_pipe_with_disk_persistence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as snapshot_dir:
            tool = create_navigation_tool(
                FakeProvider(["ok"]),
                isolated=True,
                max_steps=4,
                snapshot_dir=snapshot_dir,
            )
            opened = tool.dispatch("open", url=f"{self.base_url}/index.html")
            self.assertEqual(opened["status"], "success")
            snap = tool.dispatch("snapshot")
            self.assertEqual(snap["status"], "success")
            page_map = tool.dispatch("snapshot_map")
            self.assertEqual(page_map["found"], True)
            self.assertEqual(page_map["source"], "live")
            self.assertEqual(page_map["content"], snap["content"])
            tool.dispatch("close")

            tool2 = create_navigation_tool(
                FakeProvider(["ok"]),
                isolated=True,
                max_steps=4,
                snapshot_dir=snapshot_dir,
            )
            self.addCleanup(tool2.dispatch, "close")
            from_disk = tool2.dispatch(
                "snapshot_map", url=f"{self.base_url}/index.html"
            )
            self.assertEqual(from_disk["found"], True)
            self.assertEqual(from_disk["source"], "disk")
            self.assertEqual(from_disk["content"], snap["content"])

    def test_consent_gate_over_pipe(self):
        from src.navigation.credentials import ProxyConfig

        tool = create_navigation_tool(
            FakeProvider(["ok"]),
            isolated=True,
            max_steps=4,
            proxy=ProxyConfig(server="http://127.0.0.1:1"),
        )
        self.addCleanup(tool.dispatch, "close")
        opened = tool.dispatch("open", url=f"{self.base_url}/index.html")
        self.assertEqual(opened["status"], "error")
        self.assertEqual(opened["error"]["type"], "consent_required")
        bare = tool.dispatch("consent")
        self.assertEqual(bare["status"], "error")
        self.assertEqual(bare["error"]["type"], "consent_required")
        granted = tool.dispatch("consent", confirm=True)
        self.assertEqual(granted["status"], "success")
        self.assertTrue(granted["granted"])
        listed = tool.dispatch("list_credentials")
        self.assertEqual(listed["status"], "success")
        self.assertIn("server", listed["proxy"])
        self.assertNotIn("password", listed["proxy"])
        status = tool.dispatch("status")
        self.assertTrue(status["consent"]["required"])
        self.assertTrue(status["consent"]["granted"])

    def test_close_reaps_worker(self):
        self.client.open(url=f"{self.base_url}/index.html")
        closed = self.client.close()
        self.assertEqual(closed["status"], "success")
        self.assertFalse(self.client.is_open)

    def test_full_tool_in_isolated_mode(self):
        provider = FakeProvider(
            [
                '{"tool": "browser", "action": "open", "params": {"url": "%s/index.html"}}\n```'
                % self.base_url,
                "Tudo certo.",
            ]
        )
        tool = create_navigation_tool(provider, isolated=True, max_steps=8)
        self.addCleanup(tool.dispatch, "close")
        opened = tool.dispatch("open", url=f"{self.base_url}/index.html")
        self.assertEqual(opened["status"], "success")
        snap = tool.dispatch("snapshot")
        self.assertEqual(snap["status"], "success")
        self.assertGreater(snap["ref_count"], 0)
        status = tool.dispatch("status")
        self.assertTrue(status["open"])
        closed = tool.dispatch("close")
        self.assertEqual(closed["status"], "success")
        after = tool.dispatch("status")
        self.assertEqual(after["status"], "error")
        self.assertEqual(after["error"]["type"], "session_closed")

    def test_subagent_delegation_in_isolated_mode(self):
        provider = FakeProvider(
            [
                '{"tool": "browser", "action": "open", "params": {"url": "%s/index.html"}}\n```'
                % self.base_url,
                '{"tool": "browser", "action": "evaluate", "params": {"expression": "return document.title"}}\n```',
                "Título verificado.",
            ]
        )
        tool = create_navigation_tool(provider, isolated=True, max_steps=8)
        self.addCleanup(tool.dispatch, "close")
        result = tool.dispatch("run", request="abra e leia o título")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Título verificado.")
        self.assertIn("/index.html", result["url"])


if __name__ == "__main__":
    unittest.main()
