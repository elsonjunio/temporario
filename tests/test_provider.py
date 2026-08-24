from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from src.providers.base import BaseProvider
from src.providers.opencode import OpenCodeProvider, ProviderError


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, body: bytes = b"{}", retry_after: str | None = None):
    """Build a real ``urllib.error.HTTPError`` (so ``except`` clauses match)."""
    import urllib.error
    from email.message import Message

    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "http://provider", code, "error", headers, io.BytesIO(body)
    )


class TestOpenCodeProviderRetry(unittest.TestCase):
    def test_retries_on_timeout_then_succeeds(self):
        provider = OpenCodeProvider(api_key="k", timeout=5, attempts=3)
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[TimeoutError("t"), TimeoutError("t"), _FakeResponse(payload)],
        ):
            result = provider.infer("prompt", "config")
        self.assertEqual(result, "ok")

    def test_fails_cleanly_after_all_attempts(self):
        provider = OpenCodeProvider(api_key="k", timeout=5, attempts=2)
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[TimeoutError("t"), TimeoutError("t")],
        ):
            with self.assertRaises(ProviderError) as ctx:
                provider.infer("prompt", "config")
        self.assertIn("attempts", str(ctx.exception))


class TestTransientHTTPRetry(unittest.TestCase):
    def _provider(self, attempts: int) -> BaseProvider:
        return OpenCodeProvider(api_key="k", timeout=5, attempts=attempts)

    def test_retries_on_429_then_succeeds(self):
        provider = self._provider(3)
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[
                _http_error(429, b'{"error":"rate limited"}'),
                _FakeResponse(payload),
            ],
        ), mock.patch("src.providers.base.time.sleep") as sleep:
            result = provider.infer("prompt", "config")
        self.assertEqual(result, "ok")
        sleep.assert_called()

    def test_retries_on_500_and_gives_up_with_provider_error(self):
        provider = self._provider(2)
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[
                _http_error(500),
                _http_error(503),
            ],
        ), mock.patch("src.providers.base.time.sleep"):
            with self.assertRaises(ProviderError) as ctx:
                provider.infer("prompt", "config")
        self.assertIn("503", str(ctx.exception))
        self.assertIn("attempts", str(ctx.exception))

    def test_non_transient_http_error_is_not_retried(self):
        provider = self._provider(5)
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[_http_error(401)],
        ) as urlopen, mock.patch("src.providers.base.time.sleep") as sleep:
            with self.assertRaises(ProviderError):
                provider.infer("prompt", "config")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_retry_after_hint_is_honored(self):
        provider = self._provider(2)
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=[
                _http_error(429, retry_after="7"),
                _FakeResponse(payload),
            ],
        ), mock.patch("src.providers.base.time.sleep") as sleep:
            provider.infer("prompt", "config")
        sleep.assert_called_once_with(7.0)

    def test_class_default_attempts_bound_the_retries(self):
        provider = OpenCodeProvider(api_key="k", timeout=5)
        effects = [_http_error(429)] * provider.attempts
        with mock.patch(
            "src.providers.base.urllib.request.urlopen",
            side_effect=effects,
        ) as urlopen, mock.patch("src.providers.base.time.sleep"):
            with self.assertRaises(ProviderError):
                provider.infer("prompt", "config")
        self.assertEqual(urlopen.call_count, provider.attempts)


if __name__ == "__main__":
    unittest.main()
