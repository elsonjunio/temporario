from __future__ import annotations

import math
import os
import re
import zlib
from typing import Any

DEFAULT_TOKEN_FACTOR = 2.5
DEFAULT_THRESHOLD = 6000
DEFAULT_KEEP_RECENT = 4
DEFAULT_MAX_SUMMARY_CHARS = 2000

_SUMMARY_SYSTEM = (
    "You condense conversation and tool history into a compact summary. "
    "Preserve all facts, decisions, file paths, error details and pending "
    "tasks. Do not add commentary."
)


def _env_int(name: str, explicit: int | None, default: int) -> int:
    if explicit is not None:
        return explicit
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, explicit: float | None, default: float) -> float:
    if explicit is not None:
        return explicit
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def estimate_tokens(text: str, factor: float | None = None) -> int:
    """Rough token estimate: word count times a factor (default 2.5).

    The factor comes from the ``CONTEXT_TOKEN_FACTOR`` env var unless given
    explicitly. This is the offline fallback estimator; no model required.
    """
    f = (
        factor
        if factor is not None
        else _env_float("CONTEXT_TOKEN_FACTOR", None, DEFAULT_TOKEN_FACTOR)
    )
    return int(count_words(text) * f)


class NgramEmbedder:
    """Offline, stdlib-only character n-gram embedding (L2-normalized TF).

    Deterministic hashing (crc32) into a fixed-dimension bag of n-grams needs
    no model files and no network. Good enough for coarse relevance scoring
    between short texts.
    """

    def __init__(self, n: int = 3, dim: int = 256) -> None:
        self.n = n
        self.dim = dim

    def embed(self, text: str) -> dict[int, float]:
        cleaned = re.sub(r"\s+", " ", text.lower()).strip()
        if not cleaned:
            return {}
        if len(cleaned) < self.n:
            cleaned += " " * (self.n - len(cleaned))
        vec: dict[int, float] = {}
        for i in range(len(cleaned) - self.n + 1):
            h = zlib.crc32(cleaned[i : i + self.n].encode("utf-8")) % self.dim
            vec[h] = vec.get(h, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    @staticmethod
    def cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(i, 0.0) for i, w in a.items())


class ContextCompressor:
    """Compresses ``Memory`` context once the estimated token count reaches a
    threshold (configurable via env). Oldest entries are summarized with the
    provider; if the provider fails it degrades to relevance-based truncation
    so compression never breaks the flow."""

    def __init__(
        self,
        provider: Any,
        *,
        threshold: int | None = None,
        factor: float | None = None,
        keep_recent: int | None = None,
        max_summary_chars: int | None = None,
        embedder: NgramEmbedder | None = None,
    ) -> None:
        self.provider = provider
        self.threshold = _env_int(
            "CONTEXT_TOKEN_THRESHOLD", threshold, DEFAULT_THRESHOLD
        )
        self.factor = _env_float("CONTEXT_TOKEN_FACTOR", factor, DEFAULT_TOKEN_FACTOR)
        self.keep_recent = _env_int(
            "CONTEXT_KEEP_RECENT", keep_recent, DEFAULT_KEEP_RECENT
        )
        self.max_summary_chars = _env_int(
            "CONTEXT_MAX_SUMMARY_CHARS", max_summary_chars, DEFAULT_MAX_SUMMARY_CHARS
        )
        self.embedder = embedder if embedder is not None else NgramEmbedder()

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def estimate(self, text: str) -> int:
        return estimate_tokens(text, self.factor)

    def _history_tokens(self, history: list[dict[str, Any]]) -> int:
        return sum(self.estimate(e.get("content", "")) for e in history)

    def maybe_compress(self, memory: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        history = memory.get_history()
        if self._history_tokens(history) < self.threshold:
            return None
        return self._compress(memory, history)

    def _recent_user(self, block: list[dict[str, Any]]) -> str:
        for entry in reversed(block):
            if entry.get("role") == "user":
                return entry.get("content", "")
        return ""

    def _order_by_relevance(
        self, block: list[dict[str, Any]], reference: str
    ) -> list[dict[str, Any]]:
        if not reference:
            return block
        ref = self.embedder.embed(reference)
        return sorted(
            block,
            key=lambda e: self.embedder.cosine(
                ref, self.embedder.embed(e.get("content", ""))
            ),
            reverse=True,
        )

    def _summarize(self, block: list[dict[str, Any]]) -> str | None:
        reference = self._recent_user(block)
        ordered = self._order_by_relevance(block, reference)
        transcript = "\n\n".join(
            f"[{e.get('role')}] {e.get('content')}" for e in ordered
        )
        prompt = (
            "Condense the history below into a compact summary preserving all "
            f"facts, decisions, file paths and pending tasks. Under "
            f"{self.max_summary_chars} characters.\n\n{transcript}"
        )
        try:
            result = self.provider.infer(prompt, _SUMMARY_SYSTEM)
        except Exception:
            return None
        if not result:
            return None
        return result[: self.max_summary_chars]

    def _truncate(
        self, block: list[dict[str, Any]], recency_tokens: int
    ) -> list[dict[str, Any]]:
        budget = self.threshold - recency_tokens
        reference = self._recent_user(block)
        kept: list[dict[str, Any]] = []
        used = 0
        for entry in self._order_by_relevance(block, reference):
            cost = self.estimate(entry.get("content", ""))
            if kept and used + cost > budget:
                continue
            kept.append(entry)
            used += cost
        return kept

    def _compress(self, memory: Any, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) <= self.keep_recent:
            return {"status": "skipped", "reason": "below keep_recent"}

        keep = list(history[-self.keep_recent :])
        block = list(history[: -self.keep_recent])

        summary = self._summarize(block)
        method = "llm"
        if summary is not None:
            new_history: list[dict[str, Any]] = [
                {"role": "summary", "content": summary}
            ] + keep
        else:
            method = "truncate"
            recency_tokens = self._history_tokens(keep)
            new_history = self._truncate(block, recency_tokens) + keep

        memory.replace_history(new_history)
        return {
            "status": "compressed",
            "method": method,
            "entries_before": len(history),
            "entries_after": len(new_history),
            "tokens_before": self._history_tokens(history),
            "tokens_after": self._history_tokens(new_history),
        }


__all__ = [
    "ContextCompressor",
    "NgramEmbedder",
    "count_words",
    "estimate_tokens",
]
