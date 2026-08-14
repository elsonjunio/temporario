"""End-to-end tests for the browser navigation toolset.

Requires ``playwright`` + a Chromium binary (``python -m playwright install
chromium``). When missing, the suite is skipped instead of failing.
"""

from __future__ import annotations

import os
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
class TestBrowserToolset(unittest.TestCase):
    server: Any
    base_url: str
    controller: BrowserController

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.controller = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(self.controller.close)

    def _open(self) -> None:
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")

    def _refs(self) -> dict[str, str]:
        snap = self.controller.snapshot()
        self.assertEqual(snap["status"], "success")
        return {meta["name"]: ref for ref, meta in snap["refs"].items()}

    def test_open_page(self):
        self._open()
        info = self.controller._page_info()
        self.assertIn("/index.html", info["url"])
        self.assertEqual(
            self.controller.evaluate("return document.title")["result"], "App de Teste"
        )

    def test_snapshot_generates_stable_refs(self):
        self._open()
        snap = self.controller.snapshot()
        self.assertEqual(snap["status"], "success")
        self.assertGreater(snap["ref_count"], 0)
        content = snap["content"]
        self.assertIn("[ref=e1]", content)
        self.assertIn("button", content)
        self.assertIn('"Criar usuário"', content)
        self.assertIn('heading "Dashboard"', content)

    def test_click_by_ref_shows_panel(self):
        self._open()
        refs = self._refs()
        result = self.controller.click(ref=refs["Mostrar painel"])
        self.assertEqual(result["status"], "success")
        visible = self.controller.assert_(kind="visible", ref=refs["Mostrar painel"])
        self.assertTrue(visible["success"])
        text = self.controller.evaluate(
            "return document.getElementById('dynamic-panel').classList.contains('hidden')"
        )["result"]
        self.assertIs(text, False)

    def test_fill_form_and_submit(self):
        self._open()
        refs = self._refs()
        self.controller.fill(ref=refs["Nome"], value="Maria Souza")
        self.controller.fill(ref=refs["E-mail"], value="maria@example.com")
        self.controller.select(ref=refs["Perfil"], value="admin")
        result = self.controller.click(ref=refs["Criar usuário"])
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="text", text="Sucesso")
        text = self.controller.evaluate(
            "return document.getElementById('form-result').textContent"
        )["result"]
        self.assertIn("Sucesso: Usuário criado", text)

    def test_evaluate_javascript(self):
        self._open()
        result = self.controller.evaluate(
            "return document.querySelector('h1').innerText"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Dashboard")

    def test_capture_console_logs(self):
        self._open()
        logs = self.controller.console(clear_after=True)["logs"]
        levels = {entry["level"] for entry in logs}
        messages = " ".join(entry["message"] for entry in logs)
        self.assertIn("app iniciado", messages)
        self.assertTrue({"log", "warn", "info"} & levels)

    def test_capture_js_error(self):
        self._open()
        refs = self._refs()
        self.controller.click(ref=refs["Disparar erro JS"])
        errors = self.controller.page_errors(clear_after=True)["errors"]
        self.assertTrue(any("falha intencional" in e["message"] for e in errors))

    def test_capture_http_request(self):
        self._open()
        refs = self._refs()
        self.controller.fill(ref=refs["Nome"], value="Ana")
        self.controller.click(ref=refs["Criar usuário"])
        self.controller.wait(condition="text", text="Sucesso")
        net = self.controller.network()
        posts = [r for r in net["requests"] if r["method"] == "POST"]
        self.assertTrue(posts, net)
        self.assertTrue(any("/api/users" in r["url"] for r in posts))

    def test_capture_http_response(self):
        self._open()
        refs = self._refs()
        self.controller.fill(ref=refs["Nome"], value="Ana")
        self.controller.click(ref=refs["Criar usuário"])
        self.controller.wait(condition="text", text="Sucesso")
        net = self.controller.network()
        posts = [r for r in net["requests"] if r["method"] == "POST"]
        detail = self.controller.get_response(posts[0]["id"], full_body=True)
        self.assertEqual(detail["status"], 201)
        self.assertIn("Usuário criado", detail["response_body"] or "")

    def test_network_errors_only(self):
        self._open()
        result = self.controller.evaluate(
            "return fetch('/api/dashboard').then(r => r.status)"
        )
        self.assertEqual(result["result"], 500)
        self.controller.wait(condition="network_idle")
        net = self.controller.network(errors_only=True)
        self.assertTrue(any("/api/dashboard" in r["url"] for r in net["requests"]))

    def test_temporary_dom_modification(self):
        self._open()
        refs = self._refs()
        set_result = self.controller.set_style(
            property="color", value="rgb(1, 2, 3)", ref=refs["Dashboard"]
        )
        self.assertEqual(set_result["status"], "success")
        styles = self.controller.get_styles(ref=refs["Dashboard"])["styles"]
        self.assertEqual(styles.get("color"), "rgb(1, 2, 3)")

    def test_screenshot_viewport_and_element(self):
        self._open()
        shot = self.controller.screenshot(kind="viewport", embed=False)
        self.assertEqual(shot["status"], "success")
        self.assertGreater(shot["bytes"], 0)
        self.assertTrue(os.path.exists(shot["path"]))
        refs = self._refs()
        elem = self.controller.screenshot_element(ref=refs["Dashboard"], embed=False)
        self.assertEqual(elem["status"], "success")
        self.assertGreater(elem["bytes"], 0)

    def test_set_viewport(self):
        self._open()
        result = self.controller.set_viewport(width=375, height=812)
        self.assertEqual(result["status"], "success")
        width = self.controller.evaluate("return window.innerWidth")["result"]
        self.assertEqual(width, 375)

    def test_assertions(self):
        self._open()
        refs = self._refs()
        ok = self.controller.assert_(kind="url", expected="/index.html")
        self.assertTrue(ok["success"])
        text_ok = self.controller.assert_(
            kind="text", ref=refs["Criar usuário"], expected="Criar usuário"
        )
        self.assertTrue(text_ok["success"])
        fail = self.controller.assert_(
            kind="text", ref=refs["Criar usuário"], expected="Nunca existe"
        )
        self.assertFalse(fail["success"])
        self.assertEqual(fail["actual"], "Criar usuário")

    def test_trace_start_stop(self):
        self._open()
        start = self.controller.trace_start()
        self.assertEqual(start["status"], "success")
        self.controller.evaluate("return 1 + 1")
        stop = self.controller.trace_stop()
        self.assertEqual(stop["status"], "success")
        self.assertTrue(stop.get("path"))
        self.assertTrue(os.path.exists(stop["path"]))

    def test_invalid_reference_error(self):
        self._open()
        result = self.controller.click(ref="e999")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_reference")
        self.assertTrue(result["error"]["recoverable"])

    def test_invalid_url_rejected(self):
        result = self.controller.open("file:///etc/passwd")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "navigation_error")

    def test_cookies_and_storage_masked(self):
        self._open()
        self.controller.evaluate("document.cookie = 'session=abc123; path=/'")
        self.controller.evaluate("localStorage.setItem('token', 'supersecreto')")
        cookies = self.controller.cookies()["cookies"]
        session = next(c for c in cookies if c["name"] == "session")
        self.assertNotIn("abc123", session["value"])
        storage = self.controller.local_storage()["items"]
        token = next(i for i in storage if i["key"] == "token")
        self.assertNotIn("supersecreto", token["value"])

    def test_get_source_and_scripts(self):
        self._open()
        source = self.controller.get_source(truncate=5000)
        self.assertIn("<title>App de Teste</title>", source["source"])
        scripts = self.controller.list_scripts()["scripts"]
        self.assertTrue(any("app.js" in s["src"] for s in scripts))
        styles = self.controller.list_stylesheets()["stylesheets"]
        self.assertTrue(any("styles.css" in s["href"] for s in styles))

    def test_close_closes_session(self):
        self._open()
        result = self.controller.close()
        self.assertEqual(result["status"], "success")
        self.assertFalse(self.controller.status()["open"])
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["error"]["type"], "session_closed")


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestBrowserTabs(unittest.TestCase):
    server: Any
    base_url: str
    controller: BrowserController

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base_url = _run_server()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.controller = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(self.controller.close)

    def _open_index(self) -> None:
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")

    def test_new_tab_opens_and_becomes_active(self):
        self._open_index()
        result = self.controller.new_tab(url=f"{self.base_url}/about.html")
        self.assertEqual(result["status"], "success")
        status = self.controller.status()
        self.assertEqual(status["tabs"], 2)
        self.assertEqual(status["active_index"], 1)
        self.assertIn("/about.html", status["url"])

    def test_refs_are_scoped_per_tab(self):
        self._open_index()
        self.controller.snapshot()
        self.assertEqual(self.controller.status()["ref_count"], 14)
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        self.assertEqual(self.controller.status()["ref_count"], 0)
        self.controller.switch_tab(index=0)
        self.assertEqual(self.controller.status()["ref_count"], 14)
        self.controller.switch_tab(index=1)
        self.assertEqual(self.controller.status()["ref_count"], 0)
        self.controller.snapshot()
        self.assertGreater(self.controller.status()["ref_count"], 0)
        self.controller.switch_tab(index=0)
        self.assertEqual(self.controller.status()["ref_count"], 14)

    def test_switch_tab_by_url(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        switched = self.controller.switch_tab(url="/about.html")
        self.assertEqual(switched["status"], "success")
        self.assertEqual(switched["active_index"], 1)
        self.assertIn("/about.html", switched["url"])
        switched = self.controller.switch_tab(index=0)
        self.assertEqual(switched["active_index"], 0)
        self.assertIn("/index.html", switched["url"])

    def test_switch_tab_by_title(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        switched = self.controller.switch_tab(title="Sobre o app")
        self.assertEqual(switched["status"], "success")
        self.assertEqual(switched["active_index"], 1)

    def test_ambiguous_switch_returns_error(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/index.html")
        result = self.controller.switch_tab(url="/index.html")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")
        self.assertIn("ambiguous", result["error"]["message"])

    def test_switch_tab_out_of_range(self):
        self._open_index()
        result = self.controller.switch_tab(index=5)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_list_tabs(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        listing = self.controller.list_tabs()
        self.assertEqual(listing["status"], "success")
        self.assertEqual(listing["count"], 2)
        active = [t for t in listing["tabs"] if t["active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["index"], 1)
        urls = {t["url"] for t in listing["tabs"]}
        self.assertIn(f"{self.base_url}/index.html", urls)
        self.assertIn(f"{self.base_url}/about.html", urls)

    def test_close_tab_by_index(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        closed = self.controller.close_tab(index=1)
        self.assertEqual(closed["status"], "success")
        status = self.controller.status()
        self.assertEqual(status["tabs"], 1)
        self.assertEqual(status["active_index"], 0)
        self.assertIn("/index.html", status["url"])

    def test_close_active_tab_falls_back_to_next(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        closed = self.controller.close_tab()
        self.assertEqual(closed["status"], "success")
        status = self.controller.status()
        self.assertEqual(status["tabs"], 1)
        self.assertEqual(status["active_index"], 0)
        self.assertIn("/index.html", status["url"])

    def test_close_last_tab(self):
        self._open_index()
        closed = self.controller.close_tab()
        self.assertEqual(closed["status"], "success")
        self.assertTrue(closed.get("all_closed"))
        self.assertFalse(self.controller.status()["open"])

    def test_network_entries_tagged_by_tab(self):
        self._open_index()
        self.controller.network()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        net = self.controller.network(errors_only=False)
        about = [r for r in net["requests"] if "/about.html" in r["url"]]
        self.assertTrue(about)
        self.assertEqual(about[0]["page"], 1)

    def test_snapshot_uses_active_tab(self):
        self._open_index()
        self.controller.new_tab(url=f"{self.base_url}/about.html")
        snap = self.controller.snapshot()
        self.assertEqual(snap["status"], "success")
        self.assertIn("Sobre", snap["content"])
        self.assertNotIn("Criar usuário", snap["content"])


if __name__ == "__main__":
    unittest.main()
