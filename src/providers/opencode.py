from __future__ import annotations

from src.providers.base import BaseProvider, ProviderError

__all__ = ["OpenCodeProvider", "ProviderError"]


class OpenCodeProvider(BaseProvider):
    """Client for OpenCode Zen (OpenAI-compatible chat completions).

    Defaults to the ``big-pickle`` model. Auth is read from the
    ``OPENCODE_API_KEY`` environment variable.
    """

    label = "OpenCode Zen"
    default_base_url = "https://opencode.ai/zen/v1"
    api_key_env = "OPENCODE_API_KEY"
    default_model = "big-pickle"
    default_timeout = 300
    attempts = 3
    require_api_key = True
