"""Minimal stdlib HTTP proxy for the browser consent tests.

Supports absolute-form HTTP requests (what Chromium sends for plain http://
traffic through a proxy) and an optional ``Proxy-Authorization`` challenge:
when ``expected_auth`` is configured on the server, requests without valid
credentials get a 407 and Chromium retries with the configured credentials.

Every handled request is recorded on ``server.requests`` so tests can assert
that the proxy (and proxy auth) were actually used.
"""

from __future__ import annotations

import base64
import http.client
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence request logging
        pass

    @property
    def _expected_auth(self) -> tuple[str, str] | None:
        return getattr(self.server, "expected_auth", None)

    def _authorized(self) -> bool:
        expected = self._expected_auth
        if expected is None:
            return True
        raw = self.headers.get("Proxy-Authorization", "")
        if not raw.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(raw[6:]).decode("utf-8", "replace")
        except Exception:
            return False
        return decoded == f"{expected[0]}:{expected[1]}"

    def _record(self, authorized: bool) -> None:
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "target": self.path,
                "authorized": authorized,
                "proxy_authorization": self.headers.get("Proxy-Authorization"),
                "timestamp": time.time(),
            }
        )

    def _challenge(self) -> None:
        body = b"proxy authentication required"
        self.send_response(407)
        self.send_header("Proxy-Authenticate", 'Basic realm="proxy"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _relay(self) -> None:
        target = self.path
        if not target.lower().startswith("http://"):
            body = b"only http proxy supported"
            self.send_response(502)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        parts = urlsplit(target)
        host = parts.hostname or ""
        port = parts.port or 80
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        skipped = {
            "proxy-authorization",
            "proxy-connection",
            "connection",
            "keep-alive",
            "proxy-authenticate",
        }
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skipped}
        length = int(self.headers.get("Content-Length", 0) or 0)
        req_body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(host, port, timeout=10)
        try:
            conn.request(self.command, path, body=req_body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except Exception as exc:  # pragma: no cover - depends on network
            body = f"proxy relay failed: {exc}".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() in (
                "transfer-encoding",
                "connection",
                "proxy-connection",
                "content-length",
                "keep-alive",
            ):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        conn.close()

    def _handle(self) -> None:
        self._record(self._authorized())
        if not self._authorized():
            self._challenge()
            return
        self._relay()

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_CONNECT(self) -> None:  # not used by the http-only tests
        body = b"only http proxy supported"
        self.send_response(502)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_proxy(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    username: str | None = None,
    password: str | None = None,
) -> ThreadingHTTPServer:
    server: Any = cast(
        ThreadingHTTPServer, ThreadingHTTPServer((host, port), ProxyHandler)
    )
    server.expected_auth = (username, password) if username else None
    server.requests = []
    return server
