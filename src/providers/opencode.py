"""OpenCode Zen provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os

from .base import CappedOpenAIProvider

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "big-pickle"


class OpenCodeProvider(CappedOpenAIProvider):
    """OpenCode Zen.

    Configuration (environment variables win over config values):

    - ``OPENCODE_API_KEY`` — bearer token (required by the server)
    - ``OPENCODE_MODEL`` — model id (default ``big-pickle``)
    - ``OPENCODE_BASE_URL`` — base URL (default ``https://opencode.ai/zen/v1``)
    """

    label = "OpenCode Zen"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            model or os.getenv("OPENCODE_MODEL") or DEFAULT_MODEL,
            base_url or os.getenv("OPENCODE_BASE_URL") or DEFAULT_BASE_URL,
            api_key or os.getenv("OPENCODE_API_KEY", ""),
            reasoning_effort,
            name="opencode",
        )
