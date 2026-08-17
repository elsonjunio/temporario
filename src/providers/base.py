from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    """Base for OpenAI-compatible chat completion providers.

    Subclasses configure the environment variable names, defaults and the
    error ``label`` as class attributes. This base provides the shared HTTP
    plumbing (stdlib ``urllib``), the two-part ``infer`` prompt, response
    extraction and retries on transient network errors (timeouts/URLError).
    """

    #: Short label used in error messages (e.g. "OpenCode Zen", "LM Studio").
    label = "provider"
    #: Base URL of the OpenAI-compatible server (no trailing slash).
    default_base_url = ""
    #: Env var holding the API key (may be empty for local servers).
    api_key_env = ""
    #: Env var holding the model id (may be empty).
    model_env = ""
    #: Env var holding the server base URL (may be empty).
    base_url_env = ""
    #: Env var holding the request timeout in seconds (may be empty).
    timeout_env = ""
    #: Model used when neither an argument nor the env var is given.
    default_model = ""
    #: Request timeout in seconds.
    default_timeout = 60
    #: Attempts on transient network errors (timeouts/URLError).
    attempts = 1
    #: Whether an API key is mandatory.
    require_api_key = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        attempts: int | None = None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else self._env(self.api_key_env) or ""
        )
        self.model = (
            model
            if model is not None
            else self._env(self.model_env) or self.default_model
        )
        self.base_url = (
            base_url or self._env(self.base_url_env) or self.default_base_url
        ).rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else int(self._env(self.timeout_env) or self.default_timeout)
        )
        self.attempts = attempts if attempts is not None else self.attempts

    @staticmethod
    def _env(name: str) -> str | None:
        return os.getenv(name) if name else None

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

    def _validate(self) -> None:
        if self.require_api_key and not self.api_key:
            raise ProviderError(
                f"{self.label} requires an API key (set {self.api_key_env})."
            )
        if not self.model:
            raise ProviderError(
                f"No model selected (set {self.model_env} or pass model=...)."
            )

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers or {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"{self.label} HTTP {exc.code}: {detail}") from exc

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
        self._validate()

        body: dict[str, Any] = {"model": self.model, "messages": messages}
        body.update(settings)

        request = urllib.request.Request(
            self._chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )

        try:
            return self._send(request)
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            for attempt in range(self.attempts - 1):
                time.sleep(2 * (attempt + 1))
                try:
                    return self._send(request)
                except (TimeoutError, urllib.error.URLError) as retry_exc:
                    last_error = retry_exc
            raise ProviderError(
                f"{self.label} request failed after {self.attempts} attempts: "
                f"{last_error}"
            ) from last_error

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"{self.label} HTTP {exc.code}: {detail}") from exc

    def _log_chat(
        self, messages: list[dict[str, Any]], payload: dict[str, Any]
    ) -> None:
        """Append one raw chat exchange to the ``AGENT_LOG`` JSONL file (off
        when the env var is unset). Used to debug how a model emits tool calls."""
        path = os.getenv("AGENT_LOG")
        if not path:
            return
        record = {
            "model": self.model,
            "messages": messages,
            "response": payload,
        }
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return

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
        self._log_chat(messages, payload)

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected response from {self.label}: {payload}"
            ) from exc

        return content or ""
