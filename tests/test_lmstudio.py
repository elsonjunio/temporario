import json
import os
import unittest
from unittest import mock

from src.main import build_provider
from src.providers.lmstudio import LMStudioProvider, ProviderError
from src.providers.opencode import OpenCodeProvider


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


CHAT_PAYLOAD = '{"choices": [{"message": {"content": "olá"}}]}'
MODELS_PAYLOAD = '{"data": [{"id": "qwen-2.5-3b"}, {"id": "llama-3.2-1b"}]}'


class LMStudioProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = LMStudioProvider(model="qwen-2.5-3b")

    def _captured_request(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(CHAT_PAYLOAD)
        ) as urlopen:
            self.provider.infer("pergunta", "config")
        return urlopen.call_args[0][0]

    def test_infer_returns_content(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(CHAT_PAYLOAD)
        ) as urlopen:
            answer = self.provider.infer("pergunta", "config")
        self.assertEqual(answer, "olá")
        body = json.loads(urlopen.call_args[0][0].data)
        self.assertEqual(body["model"], "qwen-2.5-3b")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_no_auth_header_without_key(self):
        request = self._captured_request()
        self.assertNotIn("Authorization", request.headers)

    def test_auth_header_with_key(self):
        provider = LMStudioProvider(model="m", api_key="segredo")
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(CHAT_PAYLOAD)
        ) as urlopen:
            provider.infer("p", "c")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.headers["Authorization"], "Bearer segredo")

    def test_chat_without_model_raises(self):
        provider = LMStudioProvider()
        with self.assertRaises(ProviderError):
            provider.chat([{"role": "user", "content": "oi"}])

    def test_models_lists_ids(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse(MODELS_PAYLOAD)
        ) as urlopen:
            self.assertEqual(self.provider.models(), ["qwen-2.5-3b", "llama-3.2-1b"])
        self.assertIn("/models", urlopen.call_args[0][0].full_url)


class BuildProviderTest(unittest.TestCase):
    def tearDown(self):
        for var in ("AGENT_PROVIDER", "LMSTUDIO_MODEL"):
            os.environ.pop(var, None)

    def test_defaults_to_opencode(self):
        self.assertIsInstance(build_provider(), OpenCodeProvider)

    def test_lmstudio_selected(self):
        os.environ["AGENT_PROVIDER"] = "lmstudio"
        os.environ["LMSTUDIO_MODEL"] = "qwen-2.5-3b"
        self.assertIsInstance(build_provider(), LMStudioProvider)

    def test_unknown_provider_raises(self):
        os.environ["AGENT_PROVIDER"] = "nada"
        with self.assertRaises(ValueError):
            build_provider()


if __name__ == "__main__":
    unittest.main()
