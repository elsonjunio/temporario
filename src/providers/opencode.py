from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class OpenCodeProvider:
    """Client for OpenCode Zen (OpenAI-compatible chat completions).

    Defaults to the ``big-pickle`` model. Auth is read from the
    ``OPENCODE_API_KEY`` environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "big-pickle",
        base_url: str = "https://opencode.ai/zen/v1",
        timeout: int = 180,
    ):
        self.api_key = api_key or os.getenv("OPENCODE_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError(
                "OPENCODE_API_KEY is not set. Configure it to use OpenCode Zen."
            )

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        body.update(settings)

        request = urllib.request.Request(
            self._chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "opencode-agent/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenCode Zen HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"OpenCode Zen request failed: {exc.reason}") from exc

        return payload

    def infer(self, user_prompt: str, config: str, **settings: Any) -> str:
        """Send a two-part prompt: the user message plus the configuration prompt.

        ``config`` holds the assembled system/configuration prompt (built with
        ``src.utils.build_config_prompt``) and contains the tool list.
        """
        messages = [
            {"role": "system", "content": config},
            {"role": "user", "content": user_prompt},
        ]

        payload = self.chat(messages, **settings)

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected response from OpenCode Zen: {payload}"
            ) from exc

        return content or ""
