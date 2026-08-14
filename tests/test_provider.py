from __future__ import annotations

import json
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
