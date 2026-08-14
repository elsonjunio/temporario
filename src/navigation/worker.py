from __future__ import annotations

"""Browser worker process: hosts the ``BrowserController`` and serves RPC
requests over stdio as JSON-lines.

Protocol (one request per line, one response per line):

    -> {"id": 1, "method": "open", "params": {"url": "https://..."}}
    <- {"id": 1, "result": {"status": "success", ...}}
    <- {"id": 1, "error": {"type": "...", "message": "...", "recoverable": true}}

Special methods:
  - ``__init__``   controller configuration (action/nav timeouts, viewport,
                   headless, policy allow/block hosts, ...)
  - ``__ping__``   liveness probe, returns the worker pid
  - ``__exit__``   graceful shutdown (closes the browser, process exits)
  - ``close``      regular ``browser.close`` action; the worker exits after it

The controller is built lazily from the ``__init__`` config so the worker can
answer ``__ping__`` before any browser is launched.
"""

import json
import os
import sys
from typing import Any

from src.navigation.controller import BrowserController
from src.navigation.credentials import ProxyConfig, SiteCredentials
from src.navigation.errors import BrowserError
from src.navigation.security import SecurityPolicy
from src.navigation.session import BrowserSession


def _build_controller(config: dict[str, Any]) -> BrowserController:
    policy = SecurityPolicy(
        allow_hosts=config.get("allow_hosts"),
        block_hosts=config.get("block_hosts"),
        allow_evaluate=config.get("allow_evaluate"),
    )
    proxy = ProxyConfig.from_dict(config.get("proxy"))
    sites: dict[str, SiteCredentials] = {}
    raw_sites = config.get("site_credentials")
    if isinstance(raw_sites, dict):
        for origin, item in raw_sites.items():
            if isinstance(item, dict):
                creds = SiteCredentials.from_dict(str(origin), item)
                if creds is not None:
                    sites[creds.origin] = creds
    first_site = next(iter(sites.values()), None)
    http_credentials = (
        {
            "username": first_site.username,
            "password": first_site.password,
            "origin": first_site.origin,
            "send": "unauthorized",
        }
        if first_site is not None
        else None
    )
    session = BrowserSession(
        policy=policy,
        headless=config.get("headless"),
        viewport=config.get("viewport"),
        trace_dir=config.get("trace_dir"),
        video_dir=config.get("video_dir"),
        video_size=config.get("video_size"),
        proxy=proxy,
        http_credentials=http_credentials,
    )
    max_snapshot_refs = config.get("max_snapshot_refs") or 60
    screenshot_dir = config.get("screenshot_dir")
    snapshot_dir = config.get("snapshot_dir")
    return BrowserController(
        session,
        policy=policy,
        action_timeout=config.get("action_timeout"),
        nav_timeout=config.get("nav_timeout"),
        max_snapshot_refs=max_snapshot_refs,
        screenshot_dir=screenshot_dir,
        snapshot_dir=snapshot_dir,
        mutation_invalidate=config.get("mutation_invalidate"),
        proxy=proxy,
        site_credentials=sites,
    )


def _send(
    out: Any, rid: Any, *, result: Any = None, error: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {"id": rid}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    out.write(json.dumps(payload) + "\n")
    out.flush()


def main() -> int:
    stdin = sys.stdin
    stdout = sys.stdout
    config: dict[str, Any] = {}
    controller: BrowserController | None = None

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _send(
                stdout,
                None,
                error={
                    "type": "browser_error",
                    "message": f"invalid worker request: {exc}",
                    "recoverable": False,
                },
            )
            continue

        rid = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        if method == "__init__":
            config = params or {}
            _send(stdout, rid, result={"ok": True})
            continue
        if method == "__ping__":
            _send(
                stdout,
                rid,
                result={"pong": True, "pid": os.getpid(), "alive": True},
            )
            continue
        if method == "__exit__":
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass
            _send(stdout, rid, result={"bye": True})
            return 0

        try:
            if controller is None:
                controller = _build_controller(config)
            handler = getattr(controller, method, None)
            if handler is None:
                raise BrowserError(f"unknown browser method {method!r}")
            result = handler(**params)
            _send(stdout, rid, result=result)
            if method == "close":
                return 0
        except Exception as exc:
            _send(
                stdout,
                rid,
                error={
                    "type": "browser_error",
                    "message": f"worker: {exc}",
                    "recoverable": True,
                },
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
