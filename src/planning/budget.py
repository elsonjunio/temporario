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
class PlanningBudget:
    """Hard limits for one planning run.

    Every limit is enforced incrementally inside the planning loop: the planner
    stops (status ``partial``) as soon as one is exceeded, or earlier when the
    model emits a valid final plan. Values can be overridden per ``run`` call
    and via ``PLANNING_*`` environment variables.

    ``max_context_chars`` caps how much of a single discovery report / tool
    result is kept in the planning context, so evidence is consumed, not
    accumulated blindly.
    """

    max_iterations: int = 12
    max_discovery_calls: int = 5
    max_tasks: int = 12
    max_context_tokens: int = 40000
    max_context_chars: int = 4096

    def with_env(self) -> "PlanningBudget":
        return PlanningBudget(
            max_iterations=_env_int(
                "PLANNING_MAX_ITERATIONS", None, self.max_iterations
            ),
            max_discovery_calls=_env_int(
                "PLANNING_MAX_DISCOVERY_CALLS", None, self.max_discovery_calls
            ),
            max_tasks=_env_int("PLANNING_MAX_TASKS", None, self.max_tasks),
            max_context_tokens=_env_int(
                "PLANNING_MAX_CONTEXT_TOKENS", None, self.max_context_tokens
            ),
            max_context_chars=_env_int(
                "PLANNING_MAX_CONTEXT_CHARS", None, self.max_context_chars
            ),
        )

    def apply_overrides(self, **overrides: int | None) -> "PlanningBudget":
        params = {
            key: value
            for key, value in overrides.items()
            if value is not None and hasattr(self, key)
        }
        return PlanningBudget(**{**self.__dict__, **params})

    def to_dict(self) -> dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_discovery_calls": self.max_discovery_calls,
            "max_tasks": self.max_tasks,
            "max_context_tokens": self.max_context_tokens,
            "max_context_chars": self.max_context_chars,
        }


__all__ = ["PlanningBudget"]
