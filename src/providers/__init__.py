"""Custom providers for agent-mini.

Registers the ``opencode`` and ``lmstudio`` providers so the stock agent-mini
CLI can select them like any built-in provider:

    from src.providers import register
    register()

    # then agent_mini.providers.create_provider(config) knows both names.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import agent_mini.config as _config
import agent_mini.providers as _providers

from .lmstudio import LMStudioProvider
from .opencode import OpenCodeProvider
from .stackspot import StackSpotProvider

__all__ = [
    "LMStudioProvider",
    "OpenCodeProvider",
    "StackSpotProvider",
    "create_provider",
    "register",
]

OPENCODE_DEFAULTS = {
    "baseUrl": "https://opencode.ai/inference/go/v1",
    "apiKey": "",
    "model": "deepseek-v4.1-flash",
    "reasoningEffort": None,
    "maxTokens": None,
}
LMSTUDIO_DEFAULTS = {
    "baseUrl": "http://localhost:1234/v1",
    "apiKey": "",
    "model": "local-model",
    "reasoningEffort": None,
    "maxTokens": None,
}
STACKSPOT_DEFAULTS = {
    "baseUrl": "https://genai-inference-app.stackspot.com",
    "authUrl": "https://idm.stackspot.com",
    "realm": "stackspot-freemium",
    "apiKey": "",
    "model": "",
    "reasoningEffort": None,
    "knowledge": True,
    "returnKs": False,
}

_DEFAULTS = {
    "opencode": OPENCODE_DEFAULTS,
    "lmstudio": LMSTUDIO_DEFAULTS,
    "stackspot": STACKSPOT_DEFAULTS,
}
_ORIGINAL_CREATE_PROVIDER = _providers.create_provider


def _env(name: str) -> str:
    return os.getenv(name, "")


def _env_int(name: str) -> int | None:
    try:
        value = int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _bool_env(name: str, default: bool) -> bool:
    return _bool_value(os.getenv(name), default)


def create_provider(config: dict[str, Any]) -> _providers.Provider:
    """Build the provider named by ``config["provider"]``.

    Unknown names fall back to agent-mini's own factory, so ``ollama``,
    ``openai`` and ``local`` keep working.
    """
    name = config["provider"]
    if name not in _DEFAULTS:
        return _ORIGINAL_CREATE_PROVIDER(config)

    cfg = config["providers"][name]
    if name == "stackspot":
        return StackSpotProvider(
            agent_id=_env("STACKSPOT_AGENT_ID") or cfg.get("agentId") or cfg.get("model") or "",
            client_id=_env("STACKSPOT_CLIENT_ID") or cfg.get("clientId") or "",
            client_secret=_env("STACKSPOT_CLIENT_SECRET") or cfg.get("clientSecret") or "",
            realm=_env("STACKSPOT_REALM") or cfg.get("realm") or "stackspot-freemium",
            base_url=_env("STACKSPOT_BASE_URL") or cfg.get("baseUrl") or "",
            auth_url=_env("STACKSPOT_AUTH_URL") or cfg.get("authUrl") or "",
            knowledge=_bool_env("STACKSPOT_KNOWLEDGE", _bool_value(cfg.get("knowledge"), True)),
            return_ks=_bool_env("STACKSPOT_RETURN_KS", _bool_value(cfg.get("returnKs"), False)),
        )
    if name == "opencode":
        # Go serves cloud-class models only. Set the tier up front so doctor and
        # the first turn agree with model_info(); an explicit tier/context window
        # in the config still wins.
        agent_cfg = config.setdefault("agent", {})
        if not agent_cfg.get("tier") and not agent_cfg.get("contextWindow"):
            agent_cfg["tier"] = "cloud"
        return OpenCodeProvider(
            model=_env("OPENCODE_MODEL") or cfg.get("model"),
            base_url=_env("OPENCODE_BASE_URL") or cfg.get("baseUrl"),
            api_key=_env("OPENCODE_API_KEY") or cfg.get("apiKey") or _env("AGENT_MINI_API_KEY"),
            reasoning_effort=cfg.get("reasoningEffort"),
            max_tokens=_env_int("OPENCODE_MAX_TOKENS") or _positive_int(cfg.get("maxTokens")),
        )
    return LMStudioProvider(
        model=_env("LMSTUDIO_MODEL") or cfg.get("model"),
        base_url=_env("LMSTUDIO_BASE_URL") or cfg.get("baseUrl"),
        api_key=_env("LMSTUDIO_API_KEY") or cfg.get("apiKey") or _env("AGENT_MINI_API_KEY"),
        reasoning_effort=cfg.get("reasoningEffort"),
        max_tokens=_env_int("LMSTUDIO_MAX_TOKENS") or _positive_int(cfg.get("maxTokens")),
    )


def register() -> None:
    """Teach agent-mini about the custom providers (idempotent)."""
    for name, defaults in _DEFAULTS.items():
        _config.DEFAULTS["providers"].setdefault(name, dict(defaults))

    missing = tuple(name for name in _DEFAULTS if name not in _config.PROVIDERS)
    if missing:
        _config.PROVIDERS = (*_config.PROVIDERS, *missing)

    _providers.create_provider = create_provider

    # The CLI imports PROVIDERS by value; keep an already-imported module in sync.
    cli = sys.modules.get("agent_mini.cli")
    if cli is not None:
        cli.PROVIDERS = _config.PROVIDERS
