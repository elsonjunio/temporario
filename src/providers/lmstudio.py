from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class LMStudioProvider:
    """Client for LM Studio's local OpenAI-compatible server.

    Config is read from the environment (via ``.env``) but every value can be
    overridden by constructor arguments:

    - ``LMSTUDIO_BASE_URL`` — server base URL (default ``http://localhost:1234/v1``)
    - ``LMSTUDIO_MODEL`` — model id to use
    - ``LMSTUDIO_API_KEY`` — optional bearer token for servers started with an
      API key; when unset, requests carry no Authorization header
    - ``LMSTUDIO_TIMEOUT`` — request timeout in seconds (default ``60``)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("LMSTUDIO_API_KEY", "")
        )
        self.model = model or os.getenv("LMSTUDIO_MODEL", "")
        self.base_url = (
            base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
        ).rstrip("/")
        self.timeout = timeout or int(os.getenv("LMSTUDIO_TIMEOUT", "60"))

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "opencode-agent/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def models(self) -> list[str]:
        """List the model ids currently loaded in the local server."""
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"User-Agent": "opencode-agent/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"LM Studio HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"LM Studio request failed: {exc.reason}") from exc
        return [
            entry.get("id", "")
            for entry in payload.get("data", [])
            if isinstance(entry, dict)
        ]

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
        if not self.model:
            raise ProviderError(
                "No model selected. Set LMSTUDIO_MODEL or pass model=..."
            )

        body: dict[str, Any] = {"model": self.model, "messages": messages}
        body.update(settings)

        request = urllib.request.Request(
            self._chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"LM Studio request failed: {exc.reason}") from exc

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
                f"Unexpected response from LM Studio: {payload}"
            ) from exc

        return content or ""
