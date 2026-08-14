"""Tests for the navigation subagent delegation (the ``browser`` registry
entry) and its private toolset isolation."""

from __future__ import annotations

import threading
import unittest
from typing import Any

from tests.browser_fixtures.serve import serve

from src.navigation import create_navigation_tool


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


def _run_server():
    server = serve()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


class TestNavigationSubAgent(unittest.TestCase):
    server: Any
    base_url: str
    provider: FakeProvider
    tool: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.provider = FakeProvider([])
        self.tool = create_navigation_tool(self.provider, max_steps=8)
        self.addCleanup(self.tool.dispatch, "close")

    def test_run_delegates_to_subagent_and_returns_summary(self):
        self.provider.responses = ["Resumo final: página carregada."]
        self.tool.dispatch("open", url=f"{self.base_url}/index.html")
        result = self.tool.dispatch("run", request="resuma a página")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Resumo final: página carregada.")
        self.assertGreaterEqual(result["iterations"], 1)
        self.assertIn("/index.html", result["url"])

    def test_subagent_executes_browser_tool_calls(self):
        self.provider.responses = [
            '```json\n{"tool": "browser", "action": "open", "params": {"url": "%s/index.html"}}\n```'
            % self.base_url,
            '```json\n{"tool": "browser", "action": "evaluate", "params": {"expression": "return document.title"}}\n```',
            "Título verificado.",
        ]
        result = self.tool.dispatch("run", request="abra e leia o título")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Título verificado.")
        self.assertIn("/index.html", result["url"])

    def test_direct_actions_work_without_llm(self):
        opened = self.tool.dispatch("open", url=f"{self.base_url}/index.html")
        self.assertEqual(opened["status"], "success")
        snap = self.tool.dispatch("snapshot")
        self.assertEqual(snap["status"], "success")
        self.assertGreater(snap["ref_count"], 0)

    def test_internal_subagent_registry_has_no_run(self):
        from src.navigation.controller import BrowserController
        from src.navigation.tools import build_browser_spec

        controller = BrowserController()
        self.addCleanup(controller.close)
        internal = build_browser_spec(controller, run_handler=None)
        self.assertNotIn("run", internal.handlers)
        external = build_browser_spec(
            controller, run_handler=lambda **p: {"status": "success"}
        )
        self.assertIn("run", external.handlers)

    def test_run_requires_request(self):
        result = self.tool.dispatch("run")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_close_resets_session_and_subagent(self):
        self.provider.responses = ["ok"]
        self.tool.dispatch("open", url=f"{self.base_url}/index.html")
        status = self.tool.dispatch("status")
        self.assertTrue(status["open"])
        closed = self.tool.dispatch("close")
        self.assertEqual(closed["status"], "success")
        after = self.tool.dispatch("status")
        self.assertFalse(after["open"])


if __name__ == "__main__":
    unittest.main()
