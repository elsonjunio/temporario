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

__all__ = [
    "LMStudioProvider",
    "OpenCodeProvider",
    "create_provider",
    "register",
]

OPENCODE_DEFAULTS = {
    "baseUrl": "https://opencode.ai/zen/v1",
    "apiKey": "",
    "model": "big-pickle",
    "reasoningEffort": None,
}
LMSTUDIO_DEFAULTS = {
    "baseUrl": "http://localhost:1234/v1",
    "apiKey": "",
    "model": "local-model",
    "reasoningEffort": None,
}

_DEFAULTS = {"opencode": OPENCODE_DEFAULTS, "lmstudio": LMSTUDIO_DEFAULTS}
_ORIGINAL_CREATE_PROVIDER = _providers.create_provider


def _env(name: str) -> str:
    return os.getenv(name, "")


def create_provider(config: dict[str, Any]) -> _providers.Provider:
    """Build the provider named by ``config["provider"]``.

    Unknown names fall back to agent-mini's own factory, so ``ollama``,
    ``openai`` and ``local`` keep working.
    """
    name = config["provider"]
    if name not in _DEFAULTS:
        return _ORIGINAL_CREATE_PROVIDER(config)

    cfg = config["providers"][name]
    if name == "opencode":
        return OpenCodeProvider(
            model=_env("OPENCODE_MODEL") or cfg.get("model"),
            base_url=_env("OPENCODE_BASE_URL") or cfg.get("baseUrl"),
            api_key=_env("OPENCODE_API_KEY") or cfg.get("apiKey") or _env("AGENT_MINI_API_KEY"),
            reasoning_effort=cfg.get("reasoningEffort"),
        )
    return LMStudioProvider(
        model=_env("LMSTUDIO_MODEL") or cfg.get("model"),
        base_url=_env("LMSTUDIO_BASE_URL") or cfg.get("baseUrl"),
        api_key=_env("LMSTUDIO_API_KEY") or cfg.get("apiKey") or _env("AGENT_MINI_API_KEY"),
        reasoning_effort=cfg.get("reasoningEffort"),
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
