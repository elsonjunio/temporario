from __future__ import annotations

import base64
import functools
import json
import os
import re
import tempfile
import time
from typing import Any, Callable

try:
    from playwright.sync_api import Error as _PlaywrightError
    from playwright.sync_api import TimeoutError as _PlaywrightTimeoutError

    PlaywrightError: type[Exception] = _PlaywrightError
    PlaywrightTimeoutError: type[Exception] = _PlaywrightTimeoutError
except ImportError:  # pragma: no cover - depends on environment
    PlaywrightError = Exception
    PlaywrightTimeoutError = Exception

from src.navigation.credentials import (
    ProxyConfig,
    SiteCredentials,
)
from src.navigation.errors import (
    AssertionFailed,
    BrowserError,
    ConsentRequired,
    ElementNotFound,
    ElementNotVisible,
    InvalidArguments,
    InvalidReference,
    JavascriptError,
    NetworkError,
    NavigationError,
    PermissionDenied,
    SessionClosed,
    TimeoutError_,
    error_result,
)
from src.navigation.security import SecurityPolicy, mask_cookie_value
from src.navigation.session import BrowserSession
from src.navigation.snapshot import (
    build_snapshot,
    read_map_file,
    snapshot_to_dict,
    write_map_file,
)
from src.navigation.trace import extract_trace_screenshots


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _structured(method: Callable[..., Any]) -> Callable[..., Any]:
    """Boundary guard: turn any uncaught ``BrowserError`` (e.g. a closed
    session) into a structured error dict instead of an exception."""

    @functools.wraps(method)
    def wrapper(self: "BrowserController", *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except BrowserError as exc:
            return error_result(exc, operation=method.__name__)

    return wrapper


class BrowserController:
    """Low-level Playwright operations with a structured, LLM-friendly
    interface. Every public method returns a dict with at least ``status`` and,
    on failure, a categorized ``error`` payload (see ``src/navigation/errors``).

    This is deliberately NOT a 1:1 Playwright wrapper: it adds stable snapshot
    references, token budgets, redaction, timeouts and recoverable errors so an
    LLM agent can drive it deterministically.
    """

    def __init__(
        self,
        session: BrowserSession | None = None,
        policy: SecurityPolicy | None = None,
        *,
        action_timeout: float | None = None,
        nav_timeout: float | None = None,
        max_snapshot_refs: int = 60,
        screenshot_dir: str | None = None,
        snapshot_dir: str | None = None,
        upload_roots: list[str] | None = None,
        mutation_invalidate: bool | None = None,
        video_dir: str | None = None,
        video_size: dict[str, int] | None = None,
        proxy: ProxyConfig | None = None,
        site_credentials: dict[str, SiteCredentials] | None = None,
    ) -> None:
        self.proxy: ProxyConfig | None = proxy
        self.site_credentials: dict[str, SiteCredentials] = dict(site_credentials or {})
        self._consent_granted = False
        if session is None:
            session = BrowserSession(
                policy=policy,
                proxy=self.proxy,
                http_credentials=self._http_credentials_dicts(),
                video_dir=video_dir,
                video_size=video_size,
            )
        self.session = session
        self.policy = policy or self.session.policy
        self.action_timeout = (
            action_timeout
            if action_timeout is not None
            else _env_float("BROWSER_ACTION_TIMEOUT", 10.0)
        )
        self.nav_timeout = (
            nav_timeout
            if nav_timeout is not None
            else _env_float("BROWSER_NAV_TIMEOUT", 30.0)
        )
        self.max_snapshot_refs = max_snapshot_refs
        self.screenshot_dir: str = (
            screenshot_dir
            or os.getenv("BROWSER_SCREENSHOT_DIR")
            or "browser-screenshots"
        )
        self.snapshot_dir: str | None = (
            snapshot_dir or os.getenv("BROWSER_SNAPSHOT_DIR") or None
        )
        self.mutation_invalidate: bool = (
            _env_bool("BROWSER_MUTATION_INVALIDATE", True)
            if mutation_invalidate is None
            else mutation_invalidate
        )
        self._console_cursor = 0
        self._error_cursor = 0
        self._network_cursor = 0
        self._upload_roots = upload_roots or [
            os.getcwd(),
            tempfile.gettempdir(),
            "/tmp/opencode",
        ]

    # -- helpers -----------------------------------------------------------

    def _action_ms(self) -> int:
        return int(self.action_timeout * 1000)

    def _nav_ms(self) -> int:
        return int(self.nav_timeout * 1000)

    def _map_exception(self, exc: Exception) -> BrowserError:
        if isinstance(exc, PlaywrightTimeoutError):
            return TimeoutError_(f"operation timed out: {exc}", recoverable=True)
        if isinstance(exc, BrowserError):
            return exc
        message = str(exc) or exc.__class__.__name__
        lowered = message.lower()
        if "target page, context or browser has been closed" in lowered:
            return SessionClosed(message, recoverable=True)
        if "not visible" in lowered or "element is not visible" in lowered:
            return ElementNotVisible(message, recoverable=True)
        if (
            "waiting for locator" in lowered
            or "could not be found" in lowered
            or "element is not attached" in lowered
            or "no element" in lowered
            or "did not match any elements" in lowered
        ):
            return ElementNotFound(message, recoverable=True)
        if isinstance(exc, PlaywrightError):
            return BrowserError(message, recoverable=True)
        return BrowserError(message, recoverable=True)

    def _ok(self, operation: str, **extra: Any) -> dict[str, Any]:
        return {"status": "success", "operation": operation, **extra}

    def _fail(self, exc: Exception, operation: str) -> dict[str, Any]:
        mapped = self._map_exception(exc)
        return error_result(mapped, operation=operation)

    def _resolve(self, ref_or_selector: str | None) -> Any:
        """Turn a snapshot ref (``e3``) or a CSS selector into a locator.

        Refs only resolve against the current snapshot's reference table; stale
        or unknown refs raise ``InvalidReference``. Refs are automatically
        invalidated when the page changed since the snapshot (structural DOM
        mutation detected by the in-page MutationObserver, or navigation to a
        different URL) unless ``BROWSER_MUTATION_INVALIDATE=0``.
        """
        if ref_or_selector is None or ref_or_selector == "":
            raise InvalidArguments("a ref or selector is required")
        value = str(ref_or_selector).strip()
        page = self.session.require_page()
        if _is_ref(value):
            meta = self.session.get_ref(value)
            if meta is None:
                raise InvalidReference(
                    f"ref {value!r} is not in the current snapshot; take a new "
                    "snapshot to regenerate references",
                    recoverable=True,
                )
            if self.mutation_invalidate and self._refs_are_stale(page):
                raise InvalidReference(
                    f"ref {value!r} is stale: the page changed since the last "
                    "snapshot (detected DOM mutation or navigation). Take a new "
                    "snapshot.",
                    recoverable=True,
                )
            return page.locator(meta["locator"])
        return page.locator(value)

    def _refs_are_stale(self, page: Any) -> bool:
        """True when the snapshot's refs no longer reflect the live page:
        either a structural DOM mutation was detected or the URL changed."""
        if self.session.page_dirty():
            return True
        snap_url = self.session.url or ""
        if not snap_url:
            return False
        try:
            current = page.url or ""
        except Exception:
            current = ""
        return current != snap_url

    def _page_info(self) -> dict[str, Any]:
        page = self.session.require_page()
        url = title = ""
        try:
            url = page.url or ""
        except Exception:
            pass
        try:
            title = page.title() or ""
        except Exception:
            pass
        return {"url": url, "title": title}

    def _truncate(self, text: Any, limit: int) -> str:
        value = str(text)
        if len(value) <= limit:
            return value
        return f"{value[:limit]}…[truncated {len(value) - limit} chars]"

    def _serialize_value(self, value: Any, limit: int) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate(value, limit)
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
        return self._truncate(text, limit)

    def _ensure_page(self) -> None:
        if self._requires_consent() and not self._consent_granted:
            raise ConsentRequired(
                "this session is configured with a proxy and/or site "
                "credentials, but consent has not been granted. No traffic has "
                "been sent yet. Only call browser.consent (confirm=true) after "
                "a human operator explicitly approves using the configured "
                "proxy/credentials.",
                recoverable=True,
            )
        self.session._ensure_page()

    # -- consent / credentials ---------------------------------------------

    def _requires_consent(self) -> bool:
        return self.proxy is not None or bool(self.site_credentials)

    def _http_credentials_dicts(self) -> dict[str, Any] | None:
        """Playwright supports a single ``http_credentials`` object per
        context; it scopes to one origin via the ``origin`` field. The first
        configured site wins; extra origins are reported but not applied."""
        if not self.site_credentials:
            return None
        creds = next(iter(self.site_credentials.values()))
        return {
            "username": creds.username,
            "password": creds.password,
            "origin": creds.origin,
            "send": "unauthorized",
        }

    def consent(self, *, confirm: bool = False) -> dict[str, Any]:
        """Explicit human consent to use the configured proxy/credentials.

        ``confirm=true`` is mandatory: a bare ``consent`` never unlocks the
        session. Until granted, every navigation action fails with
        ``consent_required`` and no browser is launched.
        """
        if not self._requires_consent():
            return self._ok("consent", granted=True, required=False)
        if not confirm:
            return error_result(
                ConsentRequired(
                    "consent requires confirm=true; grant it only after the "
                    "human operator approves using the configured "
                    "proxy/credentials.",
                    recoverable=True,
                ),
                operation="consent",
            )
        self._consent_granted = True
        return self._ok("consent", granted=True, required=True)

    def revoke_consent(self) -> dict[str, Any]:
        was_open = self.session.is_open
        self._consent_granted = False
        if self._requires_consent() and was_open:
            self.session.close()
        return self._ok(
            "revoke_consent",
            granted=False,
            session_closed=was_open,
        )

    def list_credentials(self) -> dict[str, Any]:
        proxy = self.proxy.masked() if self.proxy is not None else None
        sites = [
            {
                "origin": creds.origin,
                "username": (f"{creds.username[:1]}******" if creds.username else None),
                "has_password": bool(creds.password),
            }
            for creds in self.site_credentials.values()
        ]
        return self._ok(
            "list_credentials",
            required=self._requires_consent(),
            consent_granted=self._consent_granted,
            proxy=proxy,
            sites=sites,
        )

    def pause(self) -> dict[str, Any]:
        """Pause headful execution and open the Playwright inspector for human
        debugging. No-op in headless mode."""
        page = self.session.require_page()
        try:
            page.pause()
        except Exception as exc:
            return self._fail(exc, "pause")
        return self._ok("pause")

    def video(self) -> dict[str, Any]:
        """Report video recording status. Recording only happens when
        ``BROWSER_VIDEO_DIR`` (or ``video_dir``) was configured before the
        browser session started. Returns ``{recording, path, finalized}`` where
        ``path`` is the active page's target file and ``finalized`` lists the
        recordings already written to disk (from closed pages)."""
        return self._ok("video", **self.session.video_info())

    def video_save(self, path: str | None = None) -> dict[str, Any]:
        """Save the finalized recordings of the session. Returns the list of
        saved files ({path, bytes, page}) plus any still-in-progress recordings
        ({path, page}). ``path``, when given, is used for the first saved file.
        Recordings that are still in progress are written automatically when
        the page or the browser closes."""
        try:
            result = self.session.video_save(path)
        except BrowserError as exc:
            return error_result(exc, operation="video_save")
        except Exception as exc:
            return self._fail(exc, "video_save")
        return self._ok("video_save", **result)

    # -- navigation --------------------------------------------------------

    def open(
        self, url: str, *, wait: str = "domcontentloaded", timeout: float | None = None
    ) -> dict[str, Any]:
        try:
            url = self.policy.validate_url(url)
        except BrowserError as exc:
            return error_result(exc, operation="open")
        self._ensure_page()
        page = self.session.page
        timeout_ms = int((timeout or self.nav_timeout) * 1000)
        try:
            page.goto(url, timeout=timeout_ms)
        except Exception as exc:
            return self._fail(exc, "open")
        try:
            if wait in ("load", "domcontentloaded", "networkidle", "commit"):
                page.wait_for_load_state(wait, timeout=timeout_ms)
        except Exception:
            pass
        return self._ok("open", **self._page_info())

    def goto(self, url: str, *, timeout: float | None = None) -> dict[str, Any]:
        return self.open(url, timeout=timeout)

    def reload(self, *, timeout: float | None = None) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            page.reload(timeout=int((timeout or self.nav_timeout) * 1000))
        except Exception as exc:
            return self._fail(exc, "reload")
        return self._ok("reload", **self._page_info())

    def back(self) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            page.go_back()
        except Exception as exc:
            return self._fail(exc, "back")
        return self._ok("back", **self._page_info())

    def forward(self) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            page.go_forward()
        except Exception as exc:
            return self._fail(exc, "forward")
        return self._ok("forward", **self._page_info())

    def wait(
        self,
        *,
        condition: str = "network_idle",
        ref: str | None = None,
        selector: str | None = None,
        text: str | None = None,
        url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        timeout_ms = int((timeout or self.action_timeout) * 1000)
        try:
            if condition == "element":
                locator = self._resolve(ref or selector)
                locator.wait_for(state="attached", timeout=timeout_ms)
            elif condition == "visible":
                locator = self._resolve(ref or selector)
                locator.wait_for(state="visible", timeout=timeout_ms)
            elif condition == "hidden":
                locator = self._resolve(ref or selector)
                locator.wait_for(state="hidden", timeout=timeout_ms)
            elif condition == "network_idle":
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            elif condition == "url":
                if not url:
                    raise InvalidArguments("wait condition 'url' requires a url")
                page.wait_for_url(url, timeout=timeout_ms)
            elif condition == "text":
                if not text:
                    raise InvalidArguments("wait condition 'text' requires a text")
                page.get_by_text(text, exact=False).first.wait_for(
                    state="visible", timeout=timeout_ms
                )
            elif condition == "timeout":
                page.wait_for_timeout(timeout_ms)
            else:
                raise InvalidArguments(f"unknown wait condition {condition!r}")
        except Exception as exc:
            return self._fail(exc, "wait")
        return self._ok("wait", condition=condition, **self._page_info())

    def close(self) -> dict[str, Any]:
        self.session.close()
        return self._ok("close")

    def status(self) -> dict[str, Any]:
        url = self.session.url or ""
        try:
            if self.session.is_open:
                url = self.session.page.url or url
        except Exception:
            pass
        return {
            "status": "success",
            "operation": "status",
            "open": self.session.is_open,
            "url": url,
            "tabs": len(self.session.pages),
            "active_index": self.session.active_index,
            "ref_count": len(self.session.refs),
            "ref_epoch": self.session.ref_epoch,
            "mutated": self.session.page_dirty(),
            "video": self.session.video_info(),
            "console_logs": len(self.session.console_logs),
            "page_errors": len(self.session.page_errors),
            "network_entries": len(self.session.network),
            "latest_snapshot": self._truncate(self.session.latest_snapshot, 2000),
            "consent": {
                "required": self._requires_consent(),
                "granted": self._consent_granted,
                "proxy": self.proxy is not None,
                "credentials": len(self.site_credentials) > 0,
            },
        }

    # -- tabs --------------------------------------------------------------

    def _select_tab(
        self,
        *,
        index: int | None = None,
        url: str | None = None,
        title: str | None = None,
    ) -> int:
        """Resolve a tab selector to an index; raises BrowserError on failure."""
        session = self.session
        session._reconcile_pages()
        if not session.pages:
            raise SessionClosed("no tabs are open", recoverable=True)
        if index is not None:
            if index < 0 or index >= len(session.pages):
                raise InvalidArguments(
                    f"tab index {index} is out of range (open tabs: {len(session.pages)})",
                    recoverable=True,
                )
            return index
        if url or title:
            matches: list[int] = []
            for i, page in enumerate(session.pages):
                page_url = page_title = ""
                try:
                    page_url = page.url or ""
                except Exception:
                    pass
                try:
                    page_title = page.title() or ""
                except Exception:
                    pass
                if url and url in page_url:
                    matches.append(i)
                elif title and title in page_title:
                    matches.append(i)
            if not matches:
                raise InvalidArguments(
                    f"no tab matches url~{url!r} title~{title!r}",
                    recoverable=True,
                )
            if len(matches) > 1:
                raise InvalidArguments(
                    f"ambiguous tab match ({len(matches)} tabs); pass an explicit "
                    "index",
                    recoverable=True,
                )
            return matches[0]
        raise InvalidArguments(
            "switch_tab requires index, url or title", recoverable=True
        )

    def new_tab(
        self, url: str | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        try:
            self._ensure_page()
            self.session.new_page()
        except BrowserError as exc:
            return error_result(exc, operation="new_tab")
        except Exception as exc:
            return self._fail(exc, "new_tab")
        if url:
            if not url.strip():
                return error_result(
                    InvalidArguments("url cannot be empty"), operation="new_tab"
                )
            try:
                url = self.policy.validate_url(url)
            except BrowserError as exc:
                return error_result(exc, operation="new_tab")
            try:
                page = self.session.require_page()
                page.goto(url, timeout=int((timeout or self.nav_timeout) * 1000))
                try:
                    page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=int((timeout or self.nav_timeout) * 1000),
                    )
                except Exception:
                    pass
            except Exception as exc:
                return self._fail(exc, "new_tab")
        return self._ok(
            "new_tab", active_index=self.session.active_index, **self._page_info()
        )

    def switch_tab(
        self,
        index: int | None = None,
        *,
        url: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        try:
            target = self._select_tab(index=index, url=url, title=title)
            self.session.switch_to(target)
        except BrowserError as exc:
            return error_result(exc, operation="switch_tab")
        except Exception as exc:
            return self._fail(exc, "switch_tab")
        return self._ok(
            "switch_tab",
            active_index=self.session.active_index,
            **self._page_info(),
        )

    def close_tab(
        self,
        index: int | None = None,
        *,
        url: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        try:
            session = self.session
            session._reconcile_pages()
            if not session.pages:
                raise SessionClosed("no tabs are open", recoverable=True)
            if index is not None:
                target = index
                if target < 0 or target >= len(session.pages):
                    raise InvalidArguments(
                        f"tab index {target} is out of range (open tabs: {len(session.pages)})",
                        recoverable=True,
                    )
            elif url or title:
                target = self._select_tab(index=None, url=url, title=title)
            else:
                target = session.active_index
            closed = session.pages[target]
            closed_url = closed.url or ""
            session.close_page(target)
        except BrowserError as exc:
            return error_result(exc, operation="close_tab")
        except Exception as exc:
            return self._fail(exc, "close_tab")
        result = self._ok("close_tab", closed_url=closed_url)
        if session.pages:
            result["active_index"] = session.active_index
            result.update(self._page_info())
        else:
            result["all_closed"] = True
        return result

    def list_tabs(self) -> dict[str, Any]:
        session = self.session
        try:
            session._reconcile_pages()
        except Exception:
            pass
        tabs: list[dict[str, Any]] = []
        for i, page in enumerate(session.pages):
            page_url = page_title = ""
            try:
                page_url = page.url or ""
            except Exception:
                pass
            try:
                page_title = page.title() or ""
            except Exception:
                pass
            meta = session._tab_meta(page)
            tabs.append(
                {
                    "index": i,
                    "active": i == session.active_index,
                    "url": page_url,
                    "title": page_title,
                    "ref_count": len(meta["refs"]),
                    "ref_epoch": meta["epoch"],
                }
            )
        return self._ok("list_tabs", count=len(tabs), tabs=tabs)

    # -- inspection --------------------------------------------------------

    def snapshot(
        self,
        *,
        max_refs: int | None = None,
        include_hidden: bool = False,
        refresh: bool = True,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            refs, text, epoch = build_snapshot(
                page,
                max_refs=max_refs or self.max_snapshot_refs,
                include_hidden=include_hidden,
            )
        except Exception as exc:
            return self._fail(exc, "snapshot")
        if refresh:
            self.session.set_snapshot_refs(
                refs, self.session.ref_epoch + 1, text, page.url or ""
            )
            self.session.clear_mutation_state()
            if self.snapshot_dir:
                title = ""
                try:
                    title = page.title() or ""
                except Exception:
                    pass
                write_map_file(
                    self.snapshot_dir,
                    snapshot_to_dict(
                        refs,
                        text,
                        page.url or "",
                        epoch=self.session.ref_epoch,
                        title=title,
                    ),
                )
        compact_refs = {
            ref: {
                "role": meta.get("role"),
                "name": meta.get("name"),
                "tag": meta.get("tag"),
                "locator": meta.get("locator"),
            }
            for ref, meta in refs.items()
        }
        return {
            "status": "success",
            "operation": "snapshot",
            "url": page.url or "",
            "epoch": self.session.ref_epoch,
            "ref_count": len(refs),
            "content": text,
            "refs": compact_refs,
        }

    def snapshot_map(
        self,
        *,
        url: str | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return the cached page map without re-crawling the DOM.

        Precedence: fresh live cache (active tab, same URL) > any tab with a
        matching URL > on-disk map (``BROWSER_SNAPSHOT_DIR``) > not_found.
        ``refresh=True`` delegates to ``snapshot`` (full re-crawl).
        """
        if refresh:
            return self.snapshot(refresh=True)

        cached: dict[str, Any] | None = None
        source = ""
        current_url = ""
        mutated = False
        live = self.session.is_open
        if live:
            try:
                current_url = self.session.page.url or ""
            except Exception:
                current_url = ""
            mutated = self.session.page_dirty()
            cached = self.session.cached_map()
            if cached and url and cached.get("url") != url:
                cached = self.session.cached_map_for(url)
            if cached is None and url:
                cached = self.session.cached_map_for(url)
            if cached is not None:
                source = "live"
        if cached is None and self.snapshot_dir and url:
            cached = read_map_file(self.snapshot_dir, url)
            if cached is not None:
                source = "disk"
        if cached is None:
            return {
                "status": "success",
                "operation": "snapshot_map",
                "found": False,
                "url": url or current_url,
                "mutated": mutated,
            }

        fresh = cached.get("url") == current_url and current_url != "" and not mutated
        compact_refs = {
            ref: {
                "role": meta.get("role"),
                "name": meta.get("name"),
                "tag": meta.get("tag"),
                "locator": meta.get("locator"),
            }
            for ref, meta in (cached.get("refs") or {}).items()
        }
        return {
            "status": "success",
            "operation": "snapshot_map",
            "found": True,
            "fresh": fresh,
            "stale": not fresh,
            "mutated": mutated,
            "source": source,
            "url": cached.get("url") or "",
            "epoch": cached.get("epoch", 0),
            "ref_count": len(compact_refs),
            "content": cached.get("content") or "",
            "refs": compact_refs,
        }

    def get_html(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if ref or selector:
                html = self._resolve(ref or selector).evaluate(
                    "(el) => el.outerHTML", None
                )
            else:
                html = page.content()
        except Exception as exc:
            return self._fail(exc, "get_html")
        return self._ok("get_html", html=self._truncate(html, 20000))

    def get_text(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if ref or selector:
                text = self._resolve(ref or selector).inner_text()
            else:
                text = page.inner_text("body")
        except Exception as exc:
            return self._fail(exc, "get_text")
        return self._ok("get_text", text=self._truncate(text, 20000))

    def get_attributes(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            attrs = self._resolve(ref or selector).evaluate("""(el) => {
                    const out = {};
                    for (const a of el.attributes) out[a.name] = a.value;
                    return out;
                }""")
        except Exception as exc:
            return self._fail(exc, "get_attributes")
        return self._ok("get_attributes", attributes=attrs or {})

    def get_styles(
        self,
        ref: str | None = None,
        selector: str | None = None,
        properties: list[str] | None = None,
    ) -> dict[str, Any]:
        props = properties or [
            "color",
            "background-color",
            "font-size",
            "font-weight",
            "display",
            "visibility",
            "opacity",
            "margin",
            "padding",
            "border",
            "position",
            "width",
            "height",
            "z-index",
            "overflow",
        ]
        script = """(props) => {
            const s = window.getComputedStyle(document.activeElement);
            return props;
        }"""
        try:
            locator = self._resolve(ref or selector)
            styles = locator.evaluate(
                """(el, props) => {
                    const s = window.getComputedStyle(el);
                    const out = {};
                    for (const p of props) out[p] = s.getPropertyValue(p);
                    return out;
                }""",
                props,
            )
        except Exception as exc:
            return self._fail(exc, "get_styles")
        return self._ok("get_styles", styles=styles or {})

    def get_bounds(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            locator = self._resolve(ref or selector)
            box = locator.bounding_box()
            visible = locator.first.is_visible()
        except Exception as exc:
            return self._fail(exc, "get_bounds")
        return self._ok(
            "get_bounds",
            visible=visible,
            bounds=box,
        )

    def _inspect_trace(self) -> dict[str, Any]:
        """Produce a diagnostic trace for the current page: ensures tracing is
        active (starting it if needed), forces at least one frame, stops the
        trace and extracts the recorded screenshots."""
        try:
            page = self.session.require_page()
        except BrowserError as exc:
            return {
                "path": None,
                "screenshots": [],
                "frames": 0,
                **error_result(exc, operation="inspect"),
            }
        if not self.session.tracing_active:
            started = self.session.trace_start()
            if started.get("status") != "success":
                return {"path": None, "screenshots": [], "frames": 0, **started}
        try:
            page.screenshot()
        except Exception:
            pass
        stopped = self.session.trace_stop()
        zip_path = stopped.get("path")
        if stopped.get("status") != "success" or not zip_path:
            return {"path": None, "screenshots": [], "frames": 0, **stopped}
        try:
            screenshots = extract_trace_screenshots(zip_path, self.session.trace_dir)
        except Exception as exc:
            return {
                "path": zip_path,
                "screenshots": [],
                "frames": 0,
                "extract_error": str(exc),
            }
        return {
            "path": zip_path,
            "screenshots": screenshots,
            "frames": len(screenshots),
        }

    def inspect(self, include: list[str] | None = None) -> dict[str, Any]:
        page = self.session.require_page()
        parts = include or [
            "url",
            "title",
            "dom",
            "console",
            "errors",
            "network",
        ]
        info: dict[str, Any] = {"status": "success", "operation": "inspect"}
        for part in parts:
            if part == "url":
                info["url"] = page.url or ""
            elif part == "title":
                info["title"] = page.title() or ""
            elif part == "dom":
                snapshot = self.snapshot(refresh=False)
                info["dom"] = self._truncate(snapshot.get("content", ""), 4000)
            elif part == "console":
                info["console"] = self._recent_console(limit=30)
            elif part == "errors":
                info["errors"] = self.page_errors(limit=30)["errors"]
            elif part == "network":
                info["network"] = [e.summary() for e in self.session.network[-20:]]
            elif part == "scripts":
                info["scripts"] = self.list_scripts()["scripts"]
            elif part == "styles":
                info["styles"] = self.list_stylesheets()["stylesheets"]
            elif part == "video":
                info["video"] = self.session.video_info()
            elif part == "trace":
                info["trace"] = self._inspect_trace()
        return info

    # -- interaction -------------------------------------------------------

    def click(
        self,
        ref: str | None = None,
        selector: str | None = None,
        *,
        wait_for_navigation: bool = False,
    ) -> dict[str, Any]:
        try:
            locator = self._resolve(ref or selector)
            locator.click(timeout=self._action_ms())
            if wait_for_navigation:
                self.session.page.wait_for_load_state("domcontentloaded")
        except Exception as exc:
            return self._fail(exc, "click")
        return self._ok("click", **self._page_info())

    def double_click(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).dblclick(timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "double_click")
        return self._ok("double_click")

    def fill(
        self,
        ref: str | None = None,
        selector: str | None = None,
        value: str = "",
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).fill(value, timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "fill")
        return self._ok("fill", value_len=len(value))

    def type(
        self,
        ref: str | None = None,
        selector: str | None = None,
        text: str = "",
        delay: float = 0.0,
    ) -> dict[str, Any]:
        try:
            locator = self._resolve(ref or selector)
            press = getattr(locator, "press_sequentially", None)
            if press is not None:
                press(text, delay=int(delay * 1000))
            else:
                locator.fill(text)
        except Exception as exc:
            return self._fail(exc, "type")
        return self._ok("type", text_len=len(text))

    def press(
        self,
        key: str = "Enter",
        ref: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if ref or selector:
                self._resolve(ref or selector).press(key, timeout=self._action_ms())
            else:
                page.keyboard.press(key)
        except Exception as exc:
            return self._fail(exc, "press")
        return self._ok("press", key=key, **self._page_info())

    def select(
        self,
        ref: str | None = None,
        selector: str | None = None,
        value: str | None = None,
        label: str | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        try:
            locator = self._resolve(ref or selector)
            if value is not None:
                selected = locator.select_option(value=value, timeout=self._action_ms())
            elif label is not None:
                selected = locator.select_option(label=label, timeout=self._action_ms())
            elif index is not None:
                selected = locator.select_option(index=index, timeout=self._action_ms())
            else:
                raise InvalidArguments("select requires value, label or index")
        except Exception as exc:
            return self._fail(exc, "select")
        return self._ok("select", selected=selected)

    def check(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).check(timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "check")
        return self._ok("check")

    def uncheck(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).uncheck(timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "uncheck")
        return self._ok("uncheck")

    def hover(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).hover(timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "hover")
        return self._ok("hover")

    def focus(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).focus()
        except Exception as exc:
            return self._fail(exc, "focus")
        return self._ok("focus")

    def scroll(
        self,
        *,
        ref: str | None = None,
        selector: str | None = None,
        direction: str | None = None,
        amount: int = 400,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if ref or selector:
                self._resolve(ref or selector).scroll_into_view_if_needed(
                    timeout=self._action_ms()
                )
                return self._ok("scroll", target="element")
            direction = (direction or "down").lower()
            delta_x = delta_y = 0
            if direction == "down":
                delta_y = amount
            elif direction == "up":
                delta_y = -amount
            elif direction == "left":
                delta_x = -amount
            elif direction == "right":
                delta_x = amount
            else:
                raise InvalidArguments(f"unknown scroll direction {direction!r}")
            page.mouse.wheel(delta_x, delta_y)
        except Exception as exc:
            return self._fail(exc, "scroll")
        return self._ok("scroll", direction=direction, amount=amount)

    def drag(
        self,
        source: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        try:
            src = self._resolve(source)
            dst = self._resolve(target)
            src.drag_to(dst, timeout=self._action_ms())
        except Exception as exc:
            return self._fail(exc, "drag")
        return self._ok("drag")

    def upload(
        self,
        paths: list[str] | None = None,
        ref: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        paths = paths or []
        allowed = [os.path.realpath(p) for p in self._upload_roots]
        verified: list[str] = []
        for raw in paths:
            real = os.path.realpath(raw)
            if not os.path.isfile(real):
                return error_result(
                    InvalidArguments(f"upload path does not exist: {raw!r}"),
                    operation="upload",
                )
            if not any(
                real == a or real.startswith(a.rstrip(os.sep) + os.sep) for a in allowed
            ):
                return error_result(
                    PermissionDenied(
                        f"upload path outside allowed roots: {raw!r}",
                        recoverable=False,
                    ),
                    operation="upload",
                )
            verified.append(real)
        try:
            self._resolve(ref or selector).set_input_files(verified)
        except Exception as exc:
            return self._fail(exc, "upload")
        return self._ok("upload", files=verified)

    # -- screenshot --------------------------------------------------------

    def _screenshot_bytes(
        self, kind: str, fmt: str, full_page: bool, quality: int | None
    ) -> bytes | None:
        page = self.session.require_page()
        if kind == "full_page":
            return page.screenshot(full_page=True, type=fmt, quality=quality)
        return page.screenshot(type=fmt, quality=quality)

    def screenshot(
        self,
        *,
        kind: str = "viewport",
        format: str = "png",
        path: str | None = None,
        quality: int | None = None,
        embed: bool = True,
        max_embed_bytes: int = 1_500_000,
    ) -> dict[str, Any]:
        fmt = (format or "png").lower()
        if fmt not in ("png", "jpeg", "webp"):
            return error_result(
                InvalidArguments(f"unsupported format {fmt!r}"), operation="screenshot"
            )
        try:
            data = self._screenshot_bytes(kind, fmt, kind == "full_page", quality)
        except Exception as exc:
            return self._fail(exc, "screenshot")
        if data is None:
            return error_result(
                BrowserError("screenshot produced no data"), operation="screenshot"
            )
        os.makedirs(self.screenshot_dir, exist_ok=True)
        final_path = path or os.path.join(
            self.screenshot_dir,
            f"{int(time.time())}.{fmt}",
        )
        try:
            with open(final_path, "wb") as fh:
                fh.write(data)
        except OSError as exc:
            return error_result(
                BrowserError(f"could not write screenshot: {exc}"),
                operation="screenshot",
            )
        result: dict[str, Any] = {
            "status": "success",
            "operation": "screenshot",
            "kind": kind,
            "format": fmt,
            "path": final_path,
            "bytes": len(data),
            **self._page_info(),
        }
        if embed and len(data) <= max_embed_bytes:
            result["data"] = base64.b64encode(data).decode("ascii")
        return result

    def screenshot_element(
        self,
        ref: str | None = None,
        selector: str | None = None,
        *,
        format: str = "png",
        path: str | None = None,
        embed: bool = True,
        max_embed_bytes: int = 1_500_000,
    ) -> dict[str, Any]:
        fmt = (format or "png").lower()
        if fmt not in ("png", "jpeg", "webp"):
            return error_result(
                InvalidArguments(f"unsupported format {fmt!r}"),
                operation="screenshot_element",
            )
        try:
            data = self._resolve(ref or selector).screenshot(type=fmt)
        except Exception as exc:
            return self._fail(exc, "screenshot_element")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        final_path = path or os.path.join(
            self.screenshot_dir, f"{int(time.time())}-element.{fmt}"
        )
        try:
            with open(final_path, "wb") as fh:
                fh.write(data)
        except OSError as exc:
            return error_result(
                BrowserError(f"could not write screenshot: {exc}"),
                operation="screenshot_element",
            )
        result: dict[str, Any] = {
            "status": "success",
            "operation": "screenshot_element",
            "format": fmt,
            "path": final_path,
            "bytes": len(data),
            **self._page_info(),
        }
        if embed and len(data) <= max_embed_bytes:
            result["data"] = base64.b64encode(data).decode("ascii")
        return result

    # -- evaluate ----------------------------------------------------------

    def evaluate(
        self,
        expression: str,
        *,
        timeout: float | None = None,
        max_chars: int = 5000,
    ) -> dict[str, Any]:
        self.policy.ensure_evaluate_allowed()
        page = self.session.require_page()
        code = (expression or "").strip()
        if not code:
            return error_result(
                InvalidArguments("expression is required"), operation="evaluate"
            )
        if code.startswith("return ") or "\n" in code:
            code = f"() => {{\n{code}\n}}"
        try:
            value = page.evaluate(code)
        except Exception as exc:
            return self._fail(JavascriptError(f"evaluate failed: {exc}"), "evaluate")
        return self._ok("evaluate", result=self._serialize_value(value, max_chars))

    # -- temporary DOM manipulation ------------------------------------------

    def set_html(
        self, html: str = "", ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if ref or selector:
                self._resolve(ref or selector).evaluate(
                    "(el, h) => { el.innerHTML = h; }", html
                )
            else:
                page.evaluate("(h) => { document.body.innerHTML = h; }", html)
        except Exception as exc:
            return self._fail(exc, "set_html")
        return self._ok("set_html")

    def set_text(
        self, text: str = "", ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, t) => { el.textContent = t; }", text
            )
        except Exception as exc:
            return self._fail(exc, "set_text")
        return self._ok("set_text")

    def set_attribute(
        self,
        name: str,
        value: str = "",
        ref: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, args) => { el.setAttribute(args[0], args[1]); }",
                [name, value],
            )
        except Exception as exc:
            return self._fail(exc, "set_attribute")
        return self._ok("set_attribute", name=name)

    def remove_attribute(
        self, name: str, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, n) => { el.removeAttribute(n); }", name
            )
        except Exception as exc:
            return self._fail(exc, "remove_attribute")
        return self._ok("remove_attribute", name=name)

    def set_style(
        self,
        property: str,
        value: str = "",
        ref: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, args) => { el.style.setProperty(args[0], args[1]); }",
                [property, value],
            )
        except Exception as exc:
            return self._fail(exc, "set_style")
        return self._ok("set_style", property=property)

    def add_class(
        self, class_name: str, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, c) => { el.classList.add(c); }", class_name
            )
        except Exception as exc:
            return self._fail(exc, "add_class")
        return self._ok("add_class", class_name=class_name)

    def remove_class(
        self, class_name: str, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate(
                "(el, c) => { el.classList.remove(c); }", class_name
            )
        except Exception as exc:
            return self._fail(exc, "remove_class")
        return self._ok("remove_class", class_name=class_name)

    def insert_html(
        self,
        html: str,
        position: str = "beforeend",
        ref: str | None = None,
        selector: str | None = None,
    ) -> dict[str, Any]:
        if position not in ("beforebegin", "afterbegin", "beforeend", "afterend"):
            return error_result(
                InvalidArguments(f"invalid position {position!r}"),
                operation="insert_html",
            )
        try:
            self._resolve(ref or selector).evaluate(
                "(el, args) => { el.insertAdjacentHTML(args[1], args[0]); }",
                [html, position],
            )
        except Exception as exc:
            return self._fail(exc, "insert_html")
        return self._ok("insert_html", position=position)

    def remove_element(
        self, ref: str | None = None, selector: str | None = None
    ) -> dict[str, Any]:
        try:
            self._resolve(ref or selector).evaluate("(el) => { el.remove(); }")
        except Exception as exc:
            return self._fail(exc, "remove_element")
        return self._ok("remove_element")

    # -- console -----------------------------------------------------------

    def _recent_console(self, limit: int = 50) -> list[dict[str, Any]]:
        logs = self.session.console_logs[self._console_cursor :]
        return logs[-limit:]

    def console(self, *, clear_after: bool = True, limit: int = 50) -> dict[str, Any]:
        logs = self._recent_console(limit=limit)
        if clear_after:
            self._console_cursor = len(self.session.console_logs)
        return self._ok(
            "console",
            count=len(logs),
            logs=logs,
        )

    def clear_console(self) -> dict[str, Any]:
        self.session.console_logs.clear()
        self._console_cursor = 0
        return self._ok("clear_console")

    def page_errors(
        self, *, clear_after: bool = False, limit: int = 50
    ) -> dict[str, Any]:
        errors = self.session.page_errors[self._error_cursor :]
        errors = errors[-limit:]
        if clear_after:
            self._error_cursor = len(self.session.page_errors)
        return self._ok("page_errors", count=len(errors), errors=errors)

    # -- network -----------------------------------------------------------

    def network(
        self,
        *,
        limit: int = 50,
        errors_only: bool = False,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        entries = self.session.network[self._network_cursor :]
        self._network_cursor = len(self.session.network)
        if errors_only:
            entries = [e for e in entries if e.error or (e.status and e.status >= 400)]
        entries = entries[-limit:]
        summary = [e.summary() for e in entries]
        if include and "method" not in include:
            summary = [{k: v for k, v in s.items() if k in include} for s in summary]
        return self._ok("network", count=len(summary), requests=summary)

    def get_request(self, id: str) -> dict[str, Any]:
        for entry in self.session.network:
            if entry.id == id:
                return self._ok("get_request", **entry.detail())
        return error_result(
            InvalidArguments(f"no network entry with id {id!r}"),
            operation="get_request",
        )

    def get_response(self, id: str, *, full_body: bool = False) -> dict[str, Any]:
        for entry in self.session.network:
            if entry.id == id:
                return self._ok("get_response", **entry.detail(full_body=full_body))
        return error_result(
            InvalidArguments(f"no network entry with id {id!r}"),
            operation="get_response",
        )

    def clear_network(self) -> dict[str, Any]:
        self.session.network.clear()
        self._network_cursor = 0
        return self._ok("clear_network")

    # -- loaded source -----------------------------------------------------

    def get_source(self, *, truncate: int = 20000) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            source = page.content()
        except Exception as exc:
            return self._fail(exc, "get_source")
        return self._ok("get_source", source=self._truncate(source, truncate))

    def list_scripts(self) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            scripts = page.evaluate("""() => {
                    const out = [];
                    const seen = new Set();
                    for (const s of document.scripts) {
                        const src = s.src || '';
                        const key = src || s.textContent.slice(0, 80);
                        if (seen.has(key)) continue;
                        seen.add(key);
                        out.push({
                            src: src,
                            type: s.type || 'classic',
                            inline: !src ? (s.textContent || '').slice(0, 500) : undefined,
                            async: s.async,
                            defer: s.defer
                        });
                    }
                    return out;
                }""")
        except Exception as exc:
            return self._fail(exc, "list_scripts")
        return self._ok("list_scripts", scripts=scripts or [])

    def get_script(self, url: str) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            result = page.evaluate(
                """(target) => {
                    for (const s of document.scripts) {
                        if (s.src === target || (s.src && s.src.endsWith(target))) {
                            return { src: s.src, content: s.textContent };
                        }
                    }
                    for (const s of document.scripts) {
                        if (!s.src) return { src: '(inline)', content: s.textContent };
                    }
                    return null;
                }""",
                url,
            )
        except Exception as exc:
            return self._fail(exc, "get_script")
        if result is None:
            return error_result(
                InvalidArguments(f"script not found: {url!r}"), operation="get_script"
            )
        result["content"] = self._truncate(result.get("content", ""), 50000)
        return self._ok("get_script", **result)

    def list_stylesheets(self) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            sheets = page.evaluate("""() => {
                    const out = [];
                    for (const link of document.querySelectorAll('link[rel="stylesheet"]')) {
                        out.push({ href: link.href, disabled: link.disabled });
                    }
                    for (const s of document.querySelectorAll('style')) {
                        out.push({ href: '(inline)', css: (s.textContent || '').slice(0, 500) });
                    }
                    return out;
                }""")
        except Exception as exc:
            return self._fail(exc, "list_stylesheets")
        return self._ok("list_stylesheets", stylesheets=sheets or [])

    # -- assertions --------------------------------------------------------

    def assert_(
        self,
        kind: str,
        *,
        ref: str | None = None,
        selector: str | None = None,
        expected: Any = None,
        attr: str | None = None,
        prop: str | None = None,
        exact: bool = False,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            if kind == "url":
                actual = page.url or ""
                matched = (actual == expected) if exact else (expected in actual)
            elif kind == "title":
                actual = page.title() or ""
                matched = (actual == expected) if exact else (expected in actual)
            elif kind == "exists":
                locator = self._resolve(ref or selector)
                count = locator.count()
                matched = count > 0
                actual = count
            elif kind == "count":
                locator = self._resolve(ref or selector)
                actual = locator.count()
                matched = int(expected) == actual
            elif kind == "visible":
                locator = self._resolve(ref or selector)
                actual = locator.first.is_visible()
                matched = bool(expected) if expected is not None else actual
            elif kind == "checked":
                locator = self._resolve(ref or selector)
                actual = locator.first.is_checked()
                matched = bool(expected) if expected is not None else actual
            elif kind == "text":
                locator = self._resolve(ref or selector)
                actual = locator.first.inner_text()
                matched = (
                    (actual.strip() == str(expected).strip())
                    if exact
                    else (str(expected) in actual)
                )
            elif kind == "attribute":
                if not attr:
                    raise InvalidArguments("assert attribute requires attr")
                locator = self._resolve(ref or selector)
                actual = locator.first.get_attribute(attr)
                matched = actual == str(expected)
            elif kind == "style":
                if not prop:
                    raise InvalidArguments("assert style requires prop")
                locator = self._resolve(ref or selector)
                actual = locator.first.evaluate(
                    "(el, p) => window.getComputedStyle(el).getPropertyValue(p)", prop
                )
                matched = (
                    (actual.strip() == str(expected).strip())
                    if exact
                    else (str(expected) in (actual or ""))
                )
            else:
                return error_result(
                    InvalidArguments(f"unknown assertion kind {kind!r}"),
                    operation="assert",
                )
        except BrowserError as exc:
            return error_result(exc, operation="assert")
        except Exception as exc:
            return self._fail(exc, "assert")

        result: dict[str, Any] = {
            "status": "success",
            "operation": "assert",
            "success": matched,
            "assertion": kind,
        }
        if ref:
            result["ref"] = ref
        if expected is not None:
            result["expected"] = expected
        result["actual"] = actual
        return result

    # -- viewport ----------------------------------------------------------

    def set_viewport(
        self,
        width: int,
        height: int,
        device_scale_factor: float | None = None,
    ) -> dict[str, Any]:
        page = self.session.require_page()
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            return error_result(
                InvalidArguments("width/height must be positive integers"),
                operation="set_viewport",
            )
        try:
            if device_scale_factor is not None:
                cdp = self.session.context.new_cdp_session(page)
                cdp.send(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": width,
                        "height": height,
                        "deviceScaleFactor": float(device_scale_factor),
                        "mobile": False,
                    },
                )
            else:
                page.set_viewport_size({"width": width, "height": height})
        except Exception as exc:
            return self._fail(exc, "set_viewport")
        return self._ok(
            "set_viewport",
            width=width,
            height=height,
            device_scale_factor=device_scale_factor,
        )

    # -- cookies & storage --------------------------------------------------

    def cookies(self, *, limit: int = 50) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            raw = self.session.context.cookies()
        except Exception as exc:
            return self._fail(exc, "cookies")
        result = []
        for cookie in raw[:limit]:
            result.append(
                {
                    "name": cookie.get("name"),
                    "value": mask_cookie_value(
                        cookie.get("name", ""), cookie.get("value")
                    ),
                    "domain": cookie.get("domain"),
                    "path": cookie.get("path"),
                    "secure": cookie.get("secure"),
                    "httpOnly": cookie.get("httpOnly"),
                }
            )
        return self._ok("cookies", count=len(result), cookies=result)

    def _storage_snapshot(self, kind: str, limit: int) -> dict[str, Any]:
        page = self.session.require_page()
        script = (
            """() => {
                const out = {};
                const n = window.localStorage.length;
                for (let i = 0; i < n; i++) {
                    const k = window.localStorage.key(i);
                    out[k] = window.localStorage.getItem(k);
                }
                return out;
            }"""
            if kind == "local"
            else """() => {
                const out = {};
                const n = window.sessionStorage.length;
                for (let i = 0; i < n; i++) {
                    const k = window.sessionStorage.key(i);
                    out[k] = window.sessionStorage.getItem(k);
                }
                return out;
            }"""
        )
        try:
            raw = page.evaluate(script)
        except Exception as exc:
            return self._fail(exc, kind)
        items = []
        for index, (key, value) in enumerate(raw.items() or []):
            if index >= limit:
                break
            masked = mask_cookie_value(key, value)
            items.append({"key": key, "value": masked})
        return self._ok(kind, count=len(items), items=items)

    def local_storage(self, *, limit: int = 50) -> dict[str, Any]:
        return self._storage_snapshot("local", limit)

    def session_storage(self, *, limit: int = 50) -> dict[str, Any]:
        return self._storage_snapshot("session", limit)

    def clear_storage(self) -> dict[str, Any]:
        page = self.session.require_page()
        try:
            self.session.context.clear_cookies()
            page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
        except Exception as exc:
            return self._fail(exc, "clear_storage")
        return self._ok("clear_storage")

    # -- trace -------------------------------------------------------------

    def trace_start(self) -> dict[str, Any]:
        return self.session.trace_start()

    def trace_stop(self) -> dict[str, Any]:
        return self.session.trace_stop()


def _is_ref(value: str) -> bool:
    return re.fullmatch(r"e\d+", value) is not None


for _name in dir(BrowserController):
    if _name.startswith("_"):
        continue
    _attr = getattr(BrowserController, _name)
    if callable(_attr):
        setattr(BrowserController, _name, _structured(_attr))
