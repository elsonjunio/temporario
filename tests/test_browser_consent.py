"""Tests for human-consent gating of proxy / site credentials.

Covered:
  - navigation is blocked (consent_required) until consent(confirm=true),
    and no browser is launched in the meantime;
  - consent without confirm=true is rejected;
  - after consent the session uses the configured proxy / http_credentials;
  - revoke_consent re-locks and closes the session;
  - list_credentials / status only expose masked data.

Requires ``playwright`` + a Chromium binary, same as ``test_browser.py``.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from tests.browser_fixtures.proxy import serve_proxy
from tests.browser_fixtures.serve import serve

from src.navigation.controller import BrowserController
from src.navigation.credentials import ProxyConfig, SiteCredentials

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


def _serve(server: Any) -> str:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}"


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestConsentGatingProxy(unittest.TestCase):
    server: Any
    proxy_server: Any
    base_url: str
    proxy_url: str
    controller: BrowserController

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = serve()
        cls.base_url = _serve(cls.server)
        cls.proxy_server = serve_proxy()
        cls.proxy_url = _serve(cls.proxy_server)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy_server.shutdown()
        cls.server.shutdown()

    def setUp(self) -> None:
        self.proxy_server.requests = []
        self.controller = BrowserController(
            action_timeout=20,
            nav_timeout=20,
            proxy=ProxyConfig(server=self.proxy_url),
        )
        self.addCleanup(self.controller.close)

    def test_open_blocked_before_consent_and_no_browser_launched(self):
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "consent_required")
        self.assertFalse(self.controller.session.is_open)
        self.assertIsNone(self.controller.session.context)

    def test_consent_without_confirm_is_rejected(self):
        result = self.controller.consent()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "consent_required")
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["error"]["type"], "consent_required")

    def test_consent_then_navigation_goes_through_proxy(self):
        result = self.controller.consent(confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["granted"])
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")
        self.assertGreater(len(self.proxy_server.requests), 0)
        targets = [r["target"] for r in self.proxy_server.requests]
        self.assertTrue(
            any(t.startswith(self.base_url) for t in targets),
            f"proxy never saw a request to {self.base_url}: {targets}",
        )
        status = self.controller.status()
        self.assertTrue(status["consent"]["granted"])

    def test_revoke_consent_closes_session_and_reblocks(self):
        self.controller.consent(confirm=True)
        self.controller.open(f"{self.base_url}/index.html")
        result = self.controller.revoke_consent()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["session_closed"])
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["error"]["type"], "consent_required")

    def test_status_and_list_credentials_are_masked(self):
        creds = self.controller.list_credentials()
        self.assertEqual(creds["status"], "success")
        self.assertTrue(creds["required"])
        self.assertFalse(creds["consent_granted"])
        self.assertIn("server", creds["proxy"])
        self.assertNotIn("password", creds["proxy"])
        status = self.controller.status()
        self.assertTrue(status["consent"]["required"])
        self.assertFalse(status["consent"]["granted"])

    def test_consent_not_required_without_config(self):
        plain = BrowserController(action_timeout=20, nav_timeout=20)
        self.addCleanup(plain.close)
        result = plain.consent()
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["required"])
        self.assertTrue(result["granted"])


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestProxyAuthentication(unittest.TestCase):
    server: Any
    proxy_server: Any
    base_url: str
    _proxy_url: str
    controller: BrowserController

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = serve()
        cls.base_url = _serve(cls.server)
        cls.proxy_server = serve_proxy(username="proxyuser", password="proxypass")
        cls._proxy_url = _serve(cls.proxy_server)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy_server.shutdown()
        cls.server.shutdown()

    def setUp(self) -> None:
        self.proxy_server.requests = []
        self.controller = BrowserController(
            action_timeout=20,
            nav_timeout=20,
            proxy=ProxyConfig(
                server=self._proxy_url,
                username="proxyuser",
                password="proxypass",
            ),
        )
        self.addCleanup(self.controller.close)

    def test_proxy_credentials_are_sent(self):
        self.controller.consent(confirm=True)
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")
        self.assertGreater(len(self.proxy_server.requests), 0)
        authorized = [r for r in self.proxy_server.requests if r["authorized"]]
        self.assertGreater(len(authorized), 0)
        header = authorized[0]["proxy_authorization"] or ""
        self.assertIn("Basic", header)


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright or chromium not available")
class TestSiteCredentialsConsent(unittest.TestCase):
    server: Any
    base_url: str
    origin: str
    controller: BrowserController

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = serve()
        cls.base_url = _serve(cls.server)
        cls.origin = cls.base_url

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        self.controller = BrowserController(
            action_timeout=20,
            nav_timeout=20,
            site_credentials={
                self.origin: SiteCredentials(
                    origin=self.origin,
                    username="api-user",
                    password="s3cret",
                )
            },
        )
        self.addCleanup(self.controller.close)

    def _fetch_status(self, controller: Any, path: str = "/api/secure") -> int:
        result = controller.evaluate(f"return fetch('{path}').then(r => r.status)")
        self.assertEqual(result["status"], "success")
        return int(result["result"])

    def test_site_credentials_unlocked_with_consent(self):
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "consent_required")
        result = self.controller.consent(confirm=True)
        self.assertEqual(result["status"], "success")
        result = self.controller.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        self.controller.wait(condition="network_idle")
        self.assertEqual(self._fetch_status(self.controller), 200)

    def test_site_credentials_are_masked_in_list_credentials(self):
        creds = self.controller.list_credentials()
        self.assertEqual(creds["status"], "success")
        self.assertTrue(creds["required"])
        self.assertEqual(len(creds["sites"]), 1)
        site = creds["sites"][0]
        self.assertEqual(site["origin"], self.origin)
        self.assertNotIn("s3cret", str(creds))
        self.assertIn("******", site["username"])

    def test_wrong_site_password_is_rejected(self):
        wrong = BrowserController(
            action_timeout=20,
            nav_timeout=20,
            site_credentials={
                self.origin: SiteCredentials(
                    origin=self.origin,
                    username="api-user",
                    password="wrong",
                )
            },
        )
        self.addCleanup(wrong.close)
        wrong.consent(confirm=True)
        result = wrong.open(f"{self.base_url}/index.html")
        self.assertEqual(result["status"], "success")
        wrong.wait(condition="network_idle")
        self.assertEqual(self._fetch_status(wrong), 401)


if __name__ == "__main__":
    unittest.main()
