from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from src.navigation.credentials import ProxyConfig
from src.navigation.errors import BrowserError, InvalidArguments, SessionClosed
from src.navigation.security import (
    SecurityPolicy,
    mask_cookie_value,
    mask_header_value,
)
from src.navigation.snapshot import (
    clear_mutation_flag,
    install_mutation_observer,
    page_is_mutated,
)

DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
MAX_BODY_CHARS = 4096
MAX_HEADER_VALUES = 512
_VIDEO_PREFIX = "page@"


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated {len(text) - limit} chars]"


def _wait_for_file(path: str, timeout: float = 5.0) -> None:
    """Best-effort: wait until a freshly-finalized video file stops growing
    (Playwright may still be flushing it after a page closes)."""
    deadline = time.time() + timeout
    last_size = -1
    stable_for = 0.0
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > 0 and size == last_size:
            stable_for += 0.25
            if stable_for >= 0.5:
                return
        else:
            stable_for = 0.0
        last_size = size
        time.sleep(0.25)


def _format_location(loc: Any) -> dict[str, Any]:
    if loc is None:
        return {}
    return {
        "line": loc.get("line"),
        "column": loc.get("column"),
    }


@dataclass
class NetworkEntry:
    id: str
    method: str
    url: str
    resource_type: str = ""
    status: int | None = None
    error: str | None = None
    page_index: int | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    post_body: str | None = None
    request_headers: dict[str, str] = field(default_factory=dict)
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: str | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.completed_at is None:
            return None
        return int((self.completed_at - self.started_at) * 1000)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "url": self.url,
            "resource_type": self.resource_type,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "page": self.page_index,
        }

    def detail(self, *, full_body: bool = False) -> dict[str, Any]:
        return {
            **self.summary(),
            "request_headers": self.request_headers,
            "response_headers": self.response_headers,
            "post_body": self.post_body,
            "response_body": (
                self.response_body if full_body else _truncate(self.response_body, 2000)
            ),
        }


class BrowserSession:
    """Owns the long-lived Playwright objects and the page-derived state.

    State kept per session (the spec's ``browser_session`` node):
      - browser / context / pages (multiple tabs; one is active),
      - snapshot references for each tab's current snapshot (``refs``) plus
        the epoch — refs are scoped per tab and follow ``switch_tab``,
      - console logs, page errors and network entries (tagged by tab index),
      - an optional trace recording.

    A session is created lazily on first use and closed explicitly with
    ``close()``. All operations go through a shared ``SecurityPolicy``.
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        *,
        headless: bool | None = None,
        viewport: dict[str, int] | None = None,
        trace_dir: str | None = None,
        video_dir: str | None = None,
        video_size: dict[str, int] | None = None,
        proxy: ProxyConfig | None = None,
        http_credentials: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        if headless is not None:
            self.headless = bool(headless)
        elif _bool_env("BROWSER_HEADED", False):
            self.headless = False
        else:
            self.headless = _bool_env("BROWSER_HEADLESS", True)
        self.viewport = viewport or dict(DEFAULT_VIEWPORT)
        self.trace_dir = trace_dir or os.getenv("BROWSER_TRACE_DIR") or "browser-traces"
        self.video_dir: str | None = video_dir or os.getenv("BROWSER_VIDEO_DIR") or None
        self.video_size: dict[str, int] = video_size or dict(DEFAULT_VIEWPORT)
        self.proxy = proxy
        self.http_credentials = http_credentials

        self._playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self.pages: list[Any] = []
        self.active_index: int = 0

        # Per-tab snapshot state, keyed by id(page).
        self._tab_refs: dict[int, dict[str, dict[str, Any]]] = {}
        self._tab_epochs: dict[int, int] = {}
        self._tab_snapshots: dict[int, str] = {}
        self._tab_urls: dict[int, str] = {}

        self.console_logs: list[dict[str, Any]] = []
        self.page_errors: list[dict[str, Any]] = []
        self.network: list[NetworkEntry] = []
        self._network_index: int = 0
        self.tracing_active = False
        self._saved_videos: set[str] = set()

    # -- per-tab state -----------------------------------------------------

    def _tab_state(self, page: Any) -> dict[str, dict[str, Any]]:
        key = id(page)
        if key not in self._tab_refs:
            self._tab_refs[key] = {}
            self._tab_epochs[key] = 0
            self._tab_snapshots[key] = ""
            self._tab_urls[key] = ""
        return self._tab_refs[key]

    def _tab_meta(self, page: Any) -> dict[str, Any]:
        self._tab_state(page)
        key = id(page)
        return {
            "refs": self._tab_refs[key],
            "epoch": self._tab_epochs[key],
            "snapshot": self._tab_snapshots[key],
            "url": self._tab_urls[key],
        }

    def _touch_tab(self, page: Any) -> None:
        key = id(page)
        self._tab_refs[key] = {}
        self._tab_epochs[key] = 0
        self._tab_snapshots[key] = ""
        self._tab_urls[key] = page.url or ""

    @property
    def refs(self) -> dict[str, dict[str, Any]]:
        return self._tab_meta(self.page)["refs"] if self.page is not None else {}

    @property
    def ref_epoch(self) -> int:
        return self._tab_meta(self.page)["epoch"] if self.page is not None else 0

    @property
    def latest_snapshot(self) -> str:
        return self._tab_meta(self.page)["snapshot"] if self.page is not None else ""

    @property
    def url(self) -> str:
        return self._tab_meta(self.page)["url"] if self.page is not None else ""

    def _page_index(self, page: Any) -> int | None:
        for index, candidate in enumerate(self.pages):
            if candidate is page:
                return index
        return None

    def _reconcile_pages(self) -> None:
        """Drop closed pages from ``self.pages``, keeping the active page in
        sync with what the browser actually has open."""
        if self.context is None:
            return
        try:
            live = [p for p in self.context.pages if not p.is_closed()]
        except Exception:
            return
        self.pages = [p for p in self.pages if any(p is q for q in live)]
        if self.page is not None and not any(self.page is p for p in self.pages):
            self.page = None
        if not self.pages:
            self.page = None
            self.active_index = 0
            return
        if self.page is None:
            self.active_index = 0
            self.page = self.pages[0]
            return
        try:
            self.active_index = self.pages.index(self.page)
        except ValueError:
            self.active_index = 0

    # -- lifecycle ---------------------------------------------------------

    def _ensure_page(self) -> None:
        if self.page is not None and not self.page.is_closed():
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on env
            raise BrowserError(
                "playwright is not installed; run `pip install playwright` "
                "and `python -m playwright install chromium`",
                recoverable=False,
            ) from exc

        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(headless=self.headless)
        context_kwargs: dict[str, Any] = {
            "viewport": self.viewport,
            "accept_downloads": True,
        }
        if self.video_dir:
            context_kwargs["record_video_dir"] = self.video_dir
            context_kwargs["record_video_size"] = self.video_size
        if self.proxy is not None:
            proxy_kwargs: dict[str, Any] = {"server": self.proxy.server}
            if self.proxy.username:
                proxy_kwargs["username"] = self.proxy.username
            if self.proxy.password:
                proxy_kwargs["password"] = self.proxy.password
            context_kwargs["proxy"] = proxy_kwargs
        if self.http_credentials:
            context_kwargs["http_credentials"] = self.http_credentials
        self.context = self.browser.new_context(**context_kwargs)
        install_mutation_observer(self.context)
        self.page = self.context.new_page()
        self.pages = [self.page]
        self.active_index = 0
        self._touch_tab(self.page)
        self._attach_page_handlers(self.page)

    def _attach_page_handlers(self, page: Any) -> None:
        if page is None:
            return
        page.on("console", lambda message: self._on_console(message, page))
        page.on("pageerror", lambda exc: self._on_page_error(exc, page))
        page.on("request", lambda request: self._on_request(request, page))
        page.on("response", lambda response: self._on_response(response, page))
        page.on("requestfailed", lambda request: self._on_request_failed(request, page))

    @property
    def is_open(self) -> bool:
        return self.page is not None and not self.page.is_closed()

    def require_page(self) -> Any:
        if not self.is_open:
            raise SessionClosed(
                "browser session is closed; call browser.open or browser.goto first",
                recoverable=True,
            )
        return self.page

    # -- tab management ----------------------------------------------------

    def new_page(self) -> Any:
        """Open a new tab and make it active."""
        if self.context is None:
            raise SessionClosed("browser session is closed", recoverable=True)
        self._reconcile_pages()
        page = self.context.new_page()
        self.pages.append(page)
        self.active_index = len(self.pages) - 1
        self.page = page
        self._touch_tab(page)
        self._attach_page_handlers(page)
        return page

    def switch_to(self, index: int) -> Any:
        self._reconcile_pages()
        if not self.pages:
            raise SessionClosed("browser session is closed", recoverable=True)
        if index < 0 or index >= len(self.pages):
            raise InvalidArguments(
                f"tab index {index} is out of range (open tabs: {len(self.pages)})",
                recoverable=True,
            )
        self.active_index = index
        self.page = self.pages[index]
        return self.page

    def close_page(self, index: int | None = None) -> Any:
        self._reconcile_pages()
        if not self.pages:
            raise SessionClosed("browser session is closed", recoverable=True)
        idx = index if index is not None else self.active_index
        if idx < 0 or idx >= len(self.pages):
            raise InvalidArguments(
                f"tab index {idx} is out of range (open tabs: {len(self.pages)})",
                recoverable=True,
            )
        closing = self.pages[idx]
        try:
            closing.close()
        except Exception:
            pass
        self.pages.pop(idx)
        self._tab_refs.pop(id(closing), None)
        self._tab_epochs.pop(id(closing), None)
        self._tab_snapshots.pop(id(closing), None)
        self._tab_urls.pop(id(closing), None)
        if self.pages:
            if self.page is closing or self.page is None or self.page.is_closed():
                self.active_index = min(idx, len(self.pages) - 1)
                self.page = self.pages[self.active_index]
            else:
                try:
                    self.active_index = self.pages.index(self.page)
                except ValueError:
                    self.active_index = 0
                    self.page = self.pages[0]
        else:
            self.page = None
            self.active_index = 0
        return closing

    def close(self) -> None:
        self.trace_stop()
        # Closing the context explicitly (before the browser) is what finalizes
        # the videos: browser.close() alone leaves them empty (0 bytes).
        if self.context is not None:
            try:
                self.context.close()
            except Exception:
                pass
        if self.browser is not None:
            try:
                self.browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.pages = []
        self.active_index = 0
        self._tab_refs.clear()
        self._tab_epochs.clear()
        self._tab_snapshots.clear()
        self._tab_urls.clear()
        self.console_logs = []
        self.page_errors = []
        self.network = []
        self._network_index = 0

    def video_info(self) -> dict[str, Any]:
        """Recording status: whether recording is enabled, the active page's
        target file, and the list of finalized recordings (auto-written when a
        page or the browser closed)."""
        if not self.video_dir:
            return {"recording": False, "path": None, "finalized": []}
        active_path: str | None = None
        try:
            video = self.page.video
            if video is not None and video.path():
                active_path = str(video.path())
        except Exception:
            pass
        finalized: list[str] = []
        try:
            for name in sorted(os.listdir(self.video_dir)):
                full = os.path.join(self.video_dir, name)
                if (
                    name.startswith(_VIDEO_PREFIX)
                    and os.path.isfile(full)
                    and full != active_path
                ):
                    finalized.append(full)
        except OSError:
            pass
        return {"recording": True, "path": active_path, "finalized": finalized}

    def video_save(self, target: str | None = None) -> dict[str, Any]:
        """Copy every finalized recording — the auto-written videos of
        already-closed pages — into the video dir (or ``target``, used for the
        first saved one). Open pages are reported as ``pending`` because their
        recordings only land on disk once the page/browser closes. Returns the
        lists ``{saved, pending}``."""
        if not self.video_dir:
            raise BrowserError(
                "video recording is not enabled; set BROWSER_VIDEO_DIR (or pass "
                "video_dir to create_navigation_tool) before opening the browser",
                recoverable=True,
            )
        try:
            os.makedirs(self.video_dir, exist_ok=True)
        except OSError as exc:
            raise BrowserError(f"could not create video dir: {exc}") from exc

        live_targets: set[str] = set()
        pending: list[dict[str, Any]] = []
        for index, page in enumerate(list(self.pages)):
            try:
                video = page.video
                if video is None or not video.path():
                    continue
                vpath = str(video.path())
            except Exception:
                continue
            live_targets.add(vpath)
            pending.append({"path": vpath, "page": index})

        sources: list[str] = []
        for name in sorted(os.listdir(self.video_dir)):
            full = os.path.join(self.video_dir, name)
            if (
                name.startswith(_VIDEO_PREFIX)
                and os.path.isfile(full)
                and full not in live_targets
            ):
                sources.append(full)

        saved: list[dict[str, Any]] = []
        for index, source in enumerate(sorted(set(sources))):
            if source in self._saved_videos:
                continue
            if target and not saved:
                dest = target
            else:
                dest = os.path.join(
                    self.video_dir, f"video-{int(time.time())}-{index}.webm"
                )
            _wait_for_file(source)
            try:
                shutil.copyfile(source, dest)
            except OSError as exc:
                raise BrowserError(f"could not copy video: {exc}") from exc
            self._saved_videos.add(source)
            try:
                size = os.path.getsize(dest)
            except OSError:
                size = 0
            saved.append({"path": dest, "bytes": size, "source": source})
        if not saved and not self._saved_videos:
            detail = f" ({len(pending)} recording in progress)" if pending else ""
            raise BrowserError(
                "no finalized video recording yet; recordings are written when "
                f"a page or the browser closes{detail}",
                recoverable=True,
            )
        return {"saved": saved, "pending": pending}

    # -- page state --------------------------------------------------------

    def set_snapshot_refs(
        self,
        refs: dict[str, dict[str, Any]],
        epoch: int,
        snapshot_text: str,
        url: str,
    ) -> None:
        if self.page is None:
            return
        key = id(self.page)
        self._tab_refs[key] = refs
        self._tab_epochs[key] = epoch
        self._tab_snapshots[key] = snapshot_text
        self._tab_urls[key] = url

    def get_ref(self, ref: str) -> dict[str, Any] | None:
        return self.refs.get(ref)

    def page_dirty(self) -> bool:
        """True when the active page's DOM changed structurally since the last
        snapshot (best-effort; False when closed or on error)."""
        if self.page is None:
            return False
        return page_is_mutated(self.page)

    def clear_mutation_state(self) -> None:
        """Reset the in-page mutation flag after a fresh snapshot."""
        if self.page is None:
            return
        clear_mutation_flag(self.page)

    def cached_map(self) -> dict[str, Any] | None:
        """The cached page map for the active tab (text + refs + url + epoch),
        or None when no snapshot was taken yet."""
        if self.page is None:
            return None
        meta = self._tab_meta(self.page)
        if not meta["snapshot"]:
            return None
        return {
            "url": meta["url"],
            "epoch": meta["epoch"],
            "refs": meta["refs"],
            "content": meta["snapshot"],
        }

    def cached_map_for(self, url: str) -> dict[str, Any] | None:
        """The cached page map for any open tab whose URL matches ``url``."""
        for page in self.pages:
            key = id(page)
            meta = self._tab_meta(page)
            if meta["snapshot"] and meta["url"] == url:
                return {
                    "url": meta["url"],
                    "epoch": meta["epoch"],
                    "refs": meta["refs"],
                    "content": meta["snapshot"],
                }
        return None

    # -- event handlers ----------------------------------------------------

    def _on_console(self, message: Any, page: Any) -> None:
        try:
            level = message.type
            text = message.text
        except Exception:
            return
        loc = _format_location(getattr(message, "location", None))
        self.console_logs.append(
            {
                "level": level,
                "message": _truncate(text, 2000),
                "source": "console-api",
                "timestamp": time.time(),
                "page": self._page_index(page),
                **loc,
            }
        )

    def _on_page_error(self, exc: Any, page: Any) -> None:
        stack = getattr(exc, "stack", None) or ""
        first_line = ""
        if stack:
            first_line = stack.strip().splitlines()[0] if stack.strip() else ""
        self.page_errors.append(
            {
                "level": "error",
                "message": _truncate(str(exc), 2000),
                "stack": _truncate(stack, 2000),
                "source": first_line,
                "timestamp": time.time(),
                "page": self._page_index(page),
            }
        )

    def _on_request(self, request: Any, page: Any) -> None:
        try:
            url = request.url
            method = request.method
            rtype = request.resource_type
        except Exception:
            return
        self._network_index += 1
        entry = NetworkEntry(
            id=f"r{self._network_index}",
            method=method,
            url=url,
            resource_type=rtype or "",
            page_index=self._page_index(page),
        )
        try:
            headers = request.headers or {}
            entry.request_headers = {
                k: (mask_header_value(k, v) or "")
                for k, v in list(headers.items())[:30]
            }
            entry.post_body = _truncate(request.post_data, MAX_BODY_CHARS)
        except Exception:
            pass
        self.network.append(entry)

    def _on_response(self, response: Any, page: Any) -> None:
        page_index = self._page_index(page)
        for entry in reversed(self.network):
            if (
                entry.url != response.url
                or entry.page_index != page_index
                or entry.status is not None
            ):
                continue
            entry.status = response.status
            entry.completed_at = time.time()
            try:
                headers = response.headers or {}
                entry.response_headers = {
                    k: (mask_header_value(k, v) or "")
                    for k, v in list(headers.items())[:30]
                }
            except Exception:
                pass
            try:
                entry.response_body = _truncate(
                    response.text()[: MAX_BODY_CHARS + 2000], MAX_BODY_CHARS
                )
            except Exception:
                entry.response_body = None
            break

    def _on_request_failed(self, request: Any, page: Any) -> None:
        page_index = self._page_index(page)
        for entry in reversed(self.network):
            if (
                entry.url != request.url
                or entry.page_index != page_index
                or entry.error is not None
            ):
                continue
            try:
                failure = getattr(request, "failure", None)
                if callable(failure):
                    failure = failure()
                entry.error = _truncate(failure or "request_failed", 500)
            except Exception:
                entry.error = "request_failed"
            entry.completed_at = time.time()
            break

    # -- trace -------------------------------------------------------------

    def trace_start(self, *, screenshots: bool = True, snapshots: bool = True) -> dict:
        self._ensure_page()
        self.context.tracing.start(screenshots=screenshots, snapshots=snapshots)
        self.tracing_active = True
        return {"status": "success", "tracing": True}

    def trace_stop(self) -> dict:
        if self.context is None:
            self.tracing_active = False
            return {"status": "success", "path": None}
        import os

        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(
            self.trace_dir,
            f"trace-{int(time.time())}.zip",
        )
        try:
            self.context.tracing.stop(path=path)
        except Exception as exc:
            return {
                "status": "error",
                "error": {
                    "type": "browser_error",
                    "message": f"trace stop failed: {exc}",
                    "recoverable": True,
                },
            }
        self.tracing_active = False
        return {"status": "success", "path": path}
