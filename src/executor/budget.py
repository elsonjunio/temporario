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
class ExecutorBudget:
    """Hard limits for one Executor run, guarding against loops.

    Every limit is enforced incrementally inside the executor: ``max_tasks``
    caps how many tasks a single ``execute`` run starts, ``max_tool_calls``
    caps the tool calls allowed for one task (the model loop stops with the
    task FAILED if it never answers), ``max_context_tokens`` caps the config
    prompt sent to the model per task, ``max_discovery_calls`` limits focused
    discovery invocations per task and ``max_discovery_context_chars`` caps how
    much of a discovery report is injected into the task's context (never the
    raw full report). ``max_retries`` is the number of repair attempts after an
    acceptance-validation failure before the task is classified FAILED /
    REPLAN_REQUIRED. ``max_validation_commands`` caps how many targeted
    verification commands one validation pass may run, ``validation_timeout``
    bounds each such command and ``max_browser_validation_steps`` bounds the
    functional browser pass. All values can be overridden per ``execute`` call
    and via ``EXECUTOR_*`` environment variables.
    """

    max_tasks: int = 20
    max_tool_calls: int = 30
    max_retries: int = 3
    max_context_tokens: int = 40000
    max_discovery_calls: int = 2
    max_discovery_context_chars: int = 4000
    max_validation_commands: int = 3
    validation_timeout: int = 120
    max_browser_validation_steps: int = 5

    def with_env(self) -> "ExecutorBudget":
        return ExecutorBudget(
            max_tasks=_env_int("EXECUTOR_MAX_TASKS", None, self.max_tasks),
            max_tool_calls=_env_int(
                "EXECUTOR_MAX_TOOL_CALLS", None, self.max_tool_calls
            ),
            max_retries=_env_int("EXECUTOR_MAX_RETRIES", None, self.max_retries),
            max_context_tokens=_env_int(
                "EXECUTOR_MAX_CONTEXT_TOKENS", None, self.max_context_tokens
            ),
            max_discovery_calls=_env_int(
                "EXECUTOR_MAX_DISCOVERY_CALLS", None, self.max_discovery_calls
            ),
            max_discovery_context_chars=_env_int(
                "EXECUTOR_MAX_DISCOVERY_CONTEXT_CHARS",
                None,
                self.max_discovery_context_chars,
            ),
            max_validation_commands=_env_int(
                "EXECUTOR_MAX_VALIDATION_COMMANDS",
                None,
                self.max_validation_commands,
            ),
            validation_timeout=_env_int(
                "EXECUTOR_VALIDATION_TIMEOUT", None, self.validation_timeout
            ),
            max_browser_validation_steps=_env_int(
                "EXECUTOR_MAX_BROWSER_VALIDATION_STEPS",
                None,
                self.max_browser_validation_steps,
            ),
        )

    def apply_overrides(self, **overrides: int | None) -> "ExecutorBudget":
        params = {
            key: value
            for key, value in overrides.items()
            if value is not None and hasattr(self, key)
        }
        return ExecutorBudget(**{**self.__dict__, **params})

    def to_dict(self) -> dict[str, int]:
        return {
            "max_tasks": self.max_tasks,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
            "max_context_tokens": self.max_context_tokens,
            "max_discovery_calls": self.max_discovery_calls,
            "max_discovery_context_chars": self.max_discovery_context_chars,
            "max_validation_commands": self.max_validation_commands,
            "validation_timeout": self.validation_timeout,
            "max_browser_validation_steps": self.max_browser_validation_steps,
        }


__all__ = ["ExecutorBudget"]
