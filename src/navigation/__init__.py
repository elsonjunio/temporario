from __future__ import annotations

import os
from typing import Any

from src.navigation.controller import BrowserController
from src.navigation.credentials import CredentialStore, ProxyConfig, SiteCredentials
from src.navigation.errors import BrowserError
from src.navigation.ipc import BrowserClient
from src.navigation.security import SecurityPolicy
from src.navigation.session import BrowserSession
from src.navigation.subagent import NavigationSubAgent
from src.navigation.tools import build_browser_spec
from src.tools.base import ToolSpec


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def create_navigation_tool(
    provider: Any,
    *,
    name: str = "browser",
    max_steps: int = 12,
    action_timeout: float | None = None,
    nav_timeout: float | None = None,
    max_snapshot_refs: int = 60,
    headless: bool | None = None,
    viewport: dict[str, int] | None = None,
    trace_dir: str | None = None,
    screenshot_dir: str | None = None,
    snapshot_dir: str | None = None,
    mutation_invalidate: bool | None = None,
    video_dir: str | None = None,
    video_size: dict[str, int] | None = None,
    policy: SecurityPolicy | None = None,
    isolated: bool | None = None,
    proxy: ProxyConfig | None = None,
    site_credentials: dict[str, SiteCredentials] | None = None,
    credentials_store: CredentialStore | None = None,
) -> ToolSpec:
    """Build the ``browser`` navigation subagent tool.

    The registry only receives this single entry. The Playwright navigation
    toolset lives in a private registry inside ``NavigationSubAgent`` and in
    the shared ``BrowserController``. Actions:

      - ``run``: delegate a natural-language navigation task to the subagent
        (discovery + planning + execution). Params: request (str), url (str,
        optional), max_steps (int, optional).
      - all direct ``browser.*`` actions (snapshot, click, fill, inspect,
        assert, screenshot, evaluate, new_tab, switch_tab, ...) execute
        immediately on the live session, letting the main agent drive the
        browser step by step.
      - ``close``: closes the session and resets the subagent context.

    ``isolated`` (default from ``BROWSER_ISOLATED`` env var, off): when true,
    the browser runs in a separate ``src.navigation.worker`` subprocess
    reached through a JSON-lines RPC ``BrowserClient``. Configuration that is
    not JSON-serializable (policy object, Playwright objects) is translated
    into the equivalent worker config.

    ``proxy`` / ``site_credentials`` configure a proxy and per-origin HTTP
    credentials. Either is applied at context creation and only after the
    human operator grants explicit consent (browser.consent, confirm=true);
    before that, navigation fails with ``consent_required`` and no browser is
    launched. When these are omitted, ``credentials_store`` (or the
    ``BROWSER_CREDENTIALS_FILE`` file, default ``.browser-credentials.json``)
    is consulted. Values are never echoed: only masked in status /
    list_credentials.

    ``snapshot_dir`` (default ``BROWSER_SNAPSHOT_DIR`` env var, opt-in)
    persists page maps to disk so ``browser.snapshot_map`` can reuse the last
    snapshot text after ``close`` or across processes — without a browser.

    ``mutation_invalidate`` (default from ``BROWSER_MUTATION_INVALIDATE`` env
    var, on) invalidates snapshot refs automatically when the page changes
    structurally (in-page MutationObserver) or navigates to a different URL.

    ``video_dir`` (default ``BROWSER_VIDEO_DIR`` env var, opt-in) records a
    video of every page in the session from context creation (Playwright cannot
    start mid-session). ``video``/``video_save`` report and save the recording;
    ``inspect(include=["trace"])`` captures a trace and extracts its
    screenshots.
    """
    policy = policy or SecurityPolicy()
    if isolated is None:
        isolated = _env_bool("BROWSER_ISOLATED", False)

    store = credentials_store or CredentialStore()
    if proxy is None:
        proxy = store.proxy()
    sites = dict(site_credentials or {})
    if not sites:
        sites = store.site_credentials()
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

    if isolated:
        controller: Any = BrowserClient(
            action_timeout=action_timeout,
            nav_timeout=nav_timeout,
            max_snapshot_refs=max_snapshot_refs,
            headless=headless,
            viewport=viewport,
            trace_dir=trace_dir,
            screenshot_dir=screenshot_dir,
            snapshot_dir=snapshot_dir,
            mutation_invalidate=mutation_invalidate,
            video_dir=video_dir,
            video_size=video_size,
            allow_hosts=policy.allow_hosts,
            block_hosts=policy.block_hosts,
            allow_evaluate=policy.allow_evaluate,
            proxy=proxy.to_dict() if proxy is not None else None,
            site_credentials=(
                {origin: creds.to_dict() for origin, creds in sites.items()} or None
            ),
        )
    else:
        session = BrowserSession(
            policy=policy,
            headless=headless,
            viewport=viewport,
            trace_dir=trace_dir,
            video_dir=video_dir,
            video_size=video_size,
            proxy=proxy,
            http_credentials=http_credentials,
        )
        controller = BrowserController(
            session,
            policy=policy,
            action_timeout=action_timeout,
            nav_timeout=nav_timeout,
            max_snapshot_refs=max_snapshot_refs,
            screenshot_dir=screenshot_dir,
            snapshot_dir=snapshot_dir,
            mutation_invalidate=mutation_invalidate,
            proxy=proxy,
            site_credentials=sites,
        )
    subagent = NavigationSubAgent(
        provider,
        controller,
        max_steps=max_steps,
    )

    def run_handler(**params: Any) -> dict:
        request = params.pop("request", "")
        try:
            return subagent.run(request, **params)
        except TypeError as exc:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": str(exc),
                    "recoverable": True,
                },
            }
        except BrowserError as exc:
            return {
                "status": "error",
                "error": exc.to_dict(),
            }

    def close_handler() -> dict:
        subagent.reset()
        return controller.close()

    spec = build_browser_spec(controller, name=name, run_handler=run_handler)
    spec.handlers["close"] = close_handler
    return spec


__all__ = [
    "BrowserClient",
    "BrowserController",
    "BrowserError",
    "BrowserSession",
    "NavigationSubAgent",
    "SecurityPolicy",
    "create_navigation_tool",
]
