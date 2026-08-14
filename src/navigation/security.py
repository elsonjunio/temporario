from __future__ import annotations

import os
from urllib.parse import urlparse

from src.navigation.errors import NavigationError, PermissionDenied

#: Sensitive header/cookie names whose values are masked when reported back
#: to the LLM context.
_SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "proxy-authorization",
    "api-key",
    "token",
}

_SENSITIVE_COOKIE_SUBSTRINGS = (
    "token",
    "session",
    "auth",
    "jwt",
    "sid",
    "secret",
    "password",
    "credential",
    "api",
)


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _host_matches(pattern: str, host: str) -> bool:
    if pattern.startswith("*."):
        suffix = pattern[1:]
        return host.endswith(suffix) or host == pattern[2:]
    return host == pattern


class SecurityPolicy:
    """URL / host restrictions and response redaction for the browser tool.

    Controlled by env vars:
      - ``BROWSER_ALLOW_HOSTS``: comma-separated allowlist (e.g.
        ``example.com,*.example.net``). Empty = allow everything valid.
      - ``BROWSER_BLOCK_HOSTS``: comma-separated blocklist; wins over allow.
      - ``BROWSER_ALLOW_EVALUATE``: ``1`` (default) lets ``browser.evaluate``
        run; ``0`` rejects it with ``permission_denied``.
    """

    def __init__(
        self,
        allow_hosts: list[str] | None = None,
        block_hosts: list[str] | None = None,
        allow_evaluate: bool | None = None,
    ) -> None:
        self.allow_hosts = (
            allow_hosts if allow_hosts is not None else _env_list("BROWSER_ALLOW_HOSTS")
        )
        self.block_hosts = (
            block_hosts if block_hosts is not None else _env_list("BROWSER_BLOCK_HOSTS")
        )
        self.allow_evaluate = (
            allow_evaluate
            if allow_evaluate is not None
            else os.getenv("BROWSER_ALLOW_EVALUATE", "1").strip().lower()
            not in ("0", "false", "no")
        )

    def validate_url(self, url: str) -> str:
        """Normalize and authorize a navigation URL.

        Raises ``NavigationError`` for malformed URLs and ``PermissionDenied``
        when the host is blocked.
        """
        if not isinstance(url, str) or not url.strip():
            raise NavigationError("empty URL")
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise NavigationError(
                f"only http(s) URLs are allowed, got {parsed.scheme or '(none)'!r}"
            )
        if not parsed.netloc:
            raise NavigationError(f"URL has no host: {url!r}")
        host = parsed.hostname or ""
        host = host.lower()
        if any(_host_matches(p, host) for p in self.block_hosts):
            raise PermissionDenied(f"host {host!r} is blocked by policy")
        if self.allow_hosts and not any(
            _host_matches(p, host) for p in self.allow_hosts
        ):
            raise PermissionDenied(f"host {host!r} is not in BROWSER_ALLOW_HOSTS")
        return url

    def ensure_evaluate_allowed(self) -> None:
        if not self.allow_evaluate:
            raise PermissionDenied(
                "browser.evaluate is disabled (BROWSER_ALLOW_EVALUATE=0)"
            )

    @staticmethod
    def is_sensitive_header(name: str) -> bool:
        return name.lower() in _SENSITIVE_HEADERS

    @staticmethod
    def is_sensitive_cookie(name: str) -> bool:
        lower = name.lower()
        return any(token in lower for token in _SENSITIVE_COOKIE_SUBSTRINGS)

    @staticmethod
    def mask(value: str, keep: int = 4) -> str:
        """Mask a sensitive value, keeping a short prefix for debugging."""
        if not value:
            return ""
        if len(value) <= keep:
            return "*" * len(value)
        return f"{value[:keep]}…{'*' * (len(value) - keep)}"


def mask_header_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if SecurityPolicy.is_sensitive_header(name):
        return SecurityPolicy.mask(value)
    return value


def mask_cookie_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if SecurityPolicy.is_sensitive_cookie(name):
        return SecurityPolicy.mask(value)
    return value
