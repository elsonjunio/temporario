import unittest

from src.context import ContextCompressor, NgramEmbedder, count_words, estimate_tokens
from src.memory import Memory


class FakeProvider:
    def __init__(self, response="SUMMARY", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        if self.error is not None:
            raise self.error
        return self.response


class FailingProvider(FakeProvider):
    pass


class TestTokenEstimate(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words("one two three"), 3)
        self.assertEqual(count_words("  spaced\tout  text\n"), 3)
        self.assertEqual(count_words(""), 0)

    def test_estimate_tokens_factor(self):
        self.assertEqual(estimate_tokens("one two three", factor=1.0), 3)
        self.assertEqual(estimate_tokens("one two three", factor=2.5), 7)
        self.assertEqual(estimate_tokens("", factor=3.0), 0)

    def test_estimate_tokens_default_factor(self):
        self.assertEqual(estimate_tokens("one two"), 5)


class TestNgramEmbedder(unittest.TestCase):
    def test_deterministic(self):
        emb = NgramEmbedder()
        self.assertEqual(emb.embed("hello world"), emb.embed("hello world"))

    def test_empty(self):
        emb = NgramEmbedder()
        self.assertEqual(emb.embed(""), {})

    def test_similar_over_dissimilar(self):
        emb = NgramEmbedder()
        ref = emb.embed("refactor the search_files module")
        similar = emb.embed("refactor search_files tool module")
        dissimilar = emb.embed("bake a chocolate birthday cake")
        self.assertGreater(emb.cosine(ref, similar), emb.cosine(ref, dissimilar))

    def test_cosine_zero(self):
        self.assertEqual(NgramEmbedder.cosine({}, {"x": 1.0}), 0.0)


class TestContextCompressor(unittest.TestCase):
    def _make(self, provider, **kwargs):
        kwargs.setdefault("threshold", 5)
        kwargs.setdefault("factor", 1.0)
        kwargs.setdefault("keep_recent", 1)
        return ContextCompressor(provider, **kwargs)

    def test_disabled_when_threshold_zero(self):
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        compressor = self._make(FakeProvider(), threshold=0)
        result = compressor.maybe_compress(memory)
        self.assertIsNone(result)
        self.assertEqual(len(memory), 2)

    def test_no_compression_below_threshold(self):
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        compressor = self._make(FakeProvider())
        self.assertIsNone(compressor.maybe_compress(memory))

    def test_compresses_via_llm(self):
        provider = FakeProvider(response="condensed history")
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        compressor = self._make(provider)
        result = compressor.maybe_compress(memory)
        self.assertEqual(result["status"], "compressed")
        self.assertEqual(result["method"], "llm")
        self.assertEqual(result["entries_before"], 2)
        self.assertEqual(result["entries_after"], 2)
        self.assertEqual(len(provider.calls), 1)
        history = memory.get_history()
        self.assertEqual(history[0]["role"], "summary")
        self.assertEqual(history[0]["content"], "condensed history")
        self.assertEqual(history[1]["content"], "delta epsilon zeta")

    def test_recency_buffer_preserved(self):
        provider = FakeProvider(response="summary")
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        memory.add("tool", "result ok done")
        compressor = self._make(provider)
        compressor.maybe_compress(memory)
        history = memory.get_history()
        self.assertEqual(history[0]["role"], "summary")
        self.assertEqual([e["content"] for e in history[1:]], ["result ok done"])

    def test_truncate_fallback_on_provider_failure(self):
        provider = FailingProvider(error=RuntimeError("boom"))
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        memory.add("tool", "result ok done")
        compressor = self._make(provider)
        result = compressor.maybe_compress(memory)
        self.assertEqual(result["status"], "compressed")
        self.assertEqual(result["method"], "truncate")
        history = memory.get_history()
        self.assertIn("result ok done", [e["content"] for e in history])
        self.assertLess(len(history), 3)

    def test_skipped_when_at_or_below_keep_recent(self):
        provider = FakeProvider(response="summary")
        memory = Memory()
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        compressor = self._make(provider, keep_recent=3)
        result = compressor.maybe_compress(memory)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(provider.calls), 0)

    def test_memory_without_compressor_unchanged(self):
        memory = Memory()
        memory.add_user("hello")
        memory.add_assistant("hi")
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory.get_history()[0]["role"], "user")

    def test_memory_with_compressor_auto_compresses(self):
        provider = FakeProvider(response="summary")
        memory = Memory(compressor=self._make(provider))
        memory.add("user", "alpha beta gamma")
        memory.add("assistant", "delta epsilon zeta")
        history = memory.get_history()
        self.assertEqual(history[0]["role"], "summary")
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
