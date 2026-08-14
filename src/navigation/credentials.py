from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.navigation.errors import BrowserError

DEFAULT_CREDENTIALS_FILE = ".browser-credentials.json"


def sanitize_proxy_server(server: str) -> str:
    """Strip any userinfo from a proxy server URL for display purposes."""
    try:
        parts = urlsplit(server)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme or 'http'}://{host}"
    except Exception:
        return server


def normalize_origin(origin: str) -> str:
    """Normalize an origin (scheme://host[:port]) for consistent keying."""
    origin = (origin or "").strip().rstrip("/")
    try:
        parts = urlsplit(origin)
        if not parts.scheme or not parts.hostname:
            return origin.lower()
        scheme = parts.scheme.lower()
        host = parts.hostname.lower()
        if parts.port:
            return f"{scheme}://{host}:{parts.port}"
        return f"{scheme}://{host}"
    except Exception:
        return origin.lower()


@dataclass
class ProxyConfig:
    server: str
    username: str | None = None
    password: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "username": self.username,
            "password": self.password,
        }

    def masked(self) -> dict[str, Any]:
        out: dict[str, Any] = {"server": sanitize_proxy_server(self.server)}
        if self.username:
            out["username"] = self.username[:1] + "*" * 6
        out["has_password"] = bool(self.password)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProxyConfig | None":
        if not isinstance(data, dict) or not data.get("server"):
            return None
        return cls(
            server=str(data["server"]),
            username=(
                str(data["username"]) if data.get("username") is not None else None
            ),
            password=(
                str(data["password"]) if data.get("password") is not None else None
            ),
        )


@dataclass
class SiteCredentials:
    origin: str
    username: str
    password: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "username": self.username,
            "password": self.password,
        }

    @classmethod
    def from_dict(
        cls, origin: str, data: dict[str, Any] | None
    ) -> "SiteCredentials | None":
        if not isinstance(data, dict) or not data.get("username"):
            return None
        return cls(
            origin=normalize_origin(origin),
            username=str(data["username"]),
            password=str(data.get("password") or ""),
        )


class CredentialStore:
    """Optional proxy/site credentials loaded from a local JSON file.

    The file is the operator's responsibility (and should be gitignored).
    Values are never echoed to the LLM: the controller only exposes presence
    and masked fields via ``status`` and ``list_credentials``.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(
            path or os.getenv("BROWSER_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
        )

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BrowserError(
                f"could not read credential store {self.path}: {exc}",
                recoverable=False,
            ) from exc
        if not isinstance(data, dict):
            raise BrowserError(
                f"credential store {self.path} must contain a JSON object",
                recoverable=False,
            )
        return data

    def proxy(self) -> ProxyConfig | None:
        return ProxyConfig.from_dict(self.load().get("proxy"))

    def site_credentials(self) -> dict[str, SiteCredentials]:
        out: dict[str, SiteCredentials] = {}
        raw = self.load().get("sites")
        if not isinstance(raw, dict):
            return out
        for origin, item in raw.items():
            if not isinstance(origin, str):
                continue
            creds = SiteCredentials.from_dict(origin, item)
            if creds is not None:
                out[creds.origin] = creds
        return out
