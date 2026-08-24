from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, explicit: int | None, default: int) -> int:
    if explicit is not None:
        return explicit
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class PatcherBudget:
    """Hard limits for one patcher run.

    Limits are enforced incrementally during the locate loop and before each
    per-file generation. Every value can be overridden per ``run`` call and
    via ``PATCHER_*`` environment variables.
    """

    max_iterations: int = 6
    max_tool_calls: int = 15
    max_context_tokens: int = 24000
    max_files_read: int = 6
    max_lines_read: int = 4000
    max_files_patched: int = 5

    def with_env(self) -> "PatcherBudget":
        return PatcherBudget(
            max_iterations=_env_int(
                "PATCHER_MAX_ITERATIONS", None, self.max_iterations
            ),
            max_tool_calls=_env_int(
                "PATCHER_MAX_TOOL_CALLS", None, self.max_tool_calls
            ),
            max_context_tokens=_env_int(
                "PATCHER_MAX_CONTEXT_TOKENS", None, self.max_context_tokens
            ),
            max_files_read=_env_int(
                "PATCHER_MAX_FILES_READ", None, self.max_files_read
            ),
            max_lines_read=_env_int(
                "PATCHER_MAX_LINES_READ", None, self.max_lines_read
            ),
            max_files_patched=_env_int(
                "PATCHER_MAX_FILES_PATCHED", None, self.max_files_patched
            ),
        )

    def apply_overrides(self, **overrides: int | None) -> "PatcherBudget":
        params = {
            key: value
            for key, value in overrides.items()
            if value is not None and hasattr(self, key)
        }
        return PatcherBudget(**{**self.__dict__, **params})

    def to_dict(self) -> dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_context_tokens": self.max_context_tokens,
            "max_files_read": self.max_files_read,
            "max_lines_read": self.max_lines_read,
            "max_files_patched": self.max_files_patched,
        }


__all__ = ["PatcherBudget"]
