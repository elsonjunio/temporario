"""Shared base for the custom OpenAI-compatible providers."""

from __future__ import annotations

import re
from typing import Any

from agent_mini.prompt import MAX_REPLY_TOKENS
from agent_mini.providers import OpenAIProvider

# OpenAI reasoning models reject max_tokens (same rule agent-mini uses for temperature).
_NO_MAX_TOKENS = re.compile(r"^(?:o\d|gpt-5)", re.IGNORECASE)


class CappedOpenAIProvider(OpenAIProvider):
    """OpenAI-compatible provider with a configurable reply-length cap.

    agent-mini only caps replies when the provider is called ``local``; the
    custom providers use their own names, so they cap explicitly here.
    ``max_tokens`` overrides agent-mini's default of 4096.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        reasoning_effort: str | None = None,
        name: str = "local",
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(model, base_url, api_key, reasoning_effort, name=name)
        try:
            max_tokens = int(max_tokens) if max_tokens is not None else 0
        except (TypeError, ValueError):
            max_tokens = 0
        self.max_tokens = max_tokens if max_tokens > 0 else MAX_REPLY_TOKENS

    def payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload = super().payload(messages, tools, temperature, stream)
        if not _NO_MAX_TOKENS.match(self.model.rsplit("/", 1)[-1]):
            payload.setdefault("max_tokens", self.max_tokens)
        return payload
