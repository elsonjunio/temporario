"""Minimal stdlib HTTP server serving the test app and mock /api/* routes.

Used by the browser tests to give the page real HTTP traffic (200/201/401/500)
without any framework dependency. Runs on an ephemeral port.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).parent / "app"

USERS = [
    {"name": "Ana Souza", "email": "ana@example.com"},
    {"name": "Bruno Lima", "email": "bruno@example.com"},
]

# Basic-auth guard for the /api/secure route.
SECURE_USERNAME = "api-user"
SECURE_PASSWORD = "s3cret"


def _json(status: int, payload: dict[str, Any]) -> tuple[bytes, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return body, "application/json; charset=utf-8"


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # silence request logging
        pass

    def _respond_json(self, status: int, payload: dict[str, Any]) -> None:
        body, ctype = _json(status, payload)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_secure(self) -> None:
        """Basic-auth protected endpoint: 401 challenge until valid creds."""
        expected = "Basic " + base64.b64encode(
            f"{SECURE_USERNAME}:{SECURE_PASSWORD}".encode("utf-8")
        ).decode("ascii")
        if self.headers.get("Authorization", "") != expected:
            body = b"authentication required"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="secure"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._respond_json(200, {"secure": "ok", "authenticated": True})

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/ping":
            return self._respond_json(200, {"status": "ok"})
        if path == "/api/users":
            return self._respond_json(200, {"users": USERS})
        if path == "/api/dashboard":
            return self._respond_json(500, {"error": "internal_error"})
        if path == "/api/secure":
            return self._handle_secure()

        rel = path.lstrip("/") or "index.html"
        target = (APP_DIR / rel).resolve()
        if not str(target).startswith(str(APP_DIR.resolve())) or not target.is_file():
            return self._respond_json(404, {"error": "not_found"})
        body = target.read_bytes()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/users":
            name = payload.get("name") or ""
            if not name.strip():
                return self._respond_json(400, {"error": "name is required"})
            return self._respond_json(
                201,
                {
                    "message": "Usuário criado",
                    "id": 123,
                    "name": name,
                },
            )
        if path == "/api/login":
            return self._respond_json(401, {"error": "invalid_credentials"})
        return self._respond_json(404, {"error": "not_found"})


def serve(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AppHandler)
    return server
