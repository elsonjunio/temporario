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
class DiscoveryBudget:
    """Hard limits for a discovery investigation.

    Every limit is enforced incrementally after each tool call; the subagent
    stops (status ``partial``) as soon as one is exceeded, or earlier when the
    model reports sufficient evidence. All values can be overridden per ``run``
    call and via ``DISCOVERY_*`` environment variables.
    """

    max_iterations: int = 10
    max_tool_calls: int = 30
    max_context_tokens: int = 40000
    max_files_read: int = 12
    max_lines_read: int = 2000

    def with_env(self) -> "DiscoveryBudget":
        return DiscoveryBudget(
            max_iterations=_env_int(
                "DISCOVERY_MAX_ITERATIONS", None, self.max_iterations
            ),
            max_tool_calls=_env_int(
                "DISCOVERY_MAX_TOOL_CALLS", None, self.max_tool_calls
            ),
            max_context_tokens=_env_int(
                "DISCOVERY_MAX_CONTEXT_TOKENS", None, self.max_context_tokens
            ),
            max_files_read=_env_int(
                "DISCOVERY_MAX_FILES_READ", None, self.max_files_read
            ),
            max_lines_read=_env_int(
                "DISCOVERY_MAX_LINES_READ", None, self.max_lines_read
            ),
        )

    def apply_overrides(self, **overrides: int | None) -> "DiscoveryBudget":
        params = {
            key: value
            for key, value in overrides.items()
            if value is not None and hasattr(self, key)
        }
        return DiscoveryBudget(**{**self.__dict__, **params})

    def to_dict(self) -> dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_context_tokens": self.max_context_tokens,
            "max_files_read": self.max_files_read,
            "max_lines_read": self.max_lines_read,
        }


__all__ = ["DiscoveryBudget"]
