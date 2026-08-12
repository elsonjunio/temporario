from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.context import ContextCompressor


@dataclass
class MemoryEntry:
    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Memory:
    """Stores conversation history and tool call results.

    Keeps an ordered list of entries and renders a compact context for the
    prompt. An optional ``ContextCompressor`` observes every add and summarizes
    old entries once the estimated token count exceeds its threshold.
    """

    def __init__(self, compressor: "ContextCompressor | None" = None) -> None:
        self._entries: list[MemoryEntry] = []
        self.compressor = compressor

    def _maybe_compress(self) -> None:
        if self.compressor is not None:
            self.compressor.maybe_compress(self)

    def add_user(self, text: str) -> None:
        self._entries.append(MemoryEntry(role="user", content=text))
        self._maybe_compress()

    def add_assistant(self, text: str) -> None:
        self._entries.append(MemoryEntry(role="assistant", content=text))
        self._maybe_compress()

    def add_tool(self, tool: str, action: str, params: dict, result: dict) -> None:
        content = f"tool={tool} action={action} params={params}\nresult={result}"
        self._entries.append(MemoryEntry(role="tool", content=content))
        self._maybe_compress()

    def add(self, role: str, content: str) -> None:
        self._entries.append(MemoryEntry(role=role, content=content))
        self._maybe_compress()

    def get_history(self) -> list[dict[str, str]]:
        return [
            {"role": entry.role, "content": entry.content, "timestamp": entry.timestamp}
            for entry in self._entries
        ]

    def get_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def replace_history(self, history: list[dict[str, Any]]) -> None:
        """Rebuild the stored entries from a list of history dicts.

        Used by the context compressor to swap old entries for a summary.
        """
        rebuilt = []
        for entry in history:
            rebuilt.append(
                MemoryEntry(
                    role=entry.get("role", ""),
                    content=entry.get("content", ""),
                    timestamp=entry.get("timestamp")
                    or datetime.now(timezone.utc).isoformat(),
                )
            )
        self._entries = rebuilt

    def get_context(self, max_entries: int | None = None) -> str:
        """Serialize recent entries into a prompt-friendly string."""
        entries = self._entries
        if max_entries is not None and max_entries >= 0:
            entries = entries[-max_entries:]

        if not entries:
            return "(empty history)"

        blocks = []
        for entry in entries:
            blocks.append(f"[{entry.role}] {entry.content}")

        return "\n".join(blocks)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
