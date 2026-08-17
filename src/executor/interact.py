from __future__ import annotations

import os
import sys
from typing import Any, Protocol, runtime_checkable

from src.executor.presenter import (
    format_confirmation,
    format_plan_preview,
    format_question,
)


@runtime_checkable
class ExecutorInteractor(Protocol):
    """Interface the Executor uses to reach the human.

    The CLI implementation prompts on the terminal; tests use scripted fakes.
    When no interactor is injected, the Executor pauses (NEED_* states) and
    returns an ``awaiting_confirmation`` payload for the caller to relay.
    """

    def confirm_plan(self, preview: dict[str, Any]) -> bool: ...

    def confirm_task(
        self, task_id: str, level: str, message: str, affected: list[str]
    ) -> bool: ...

    def ask(
        self,
        task_id: str,
        question: str,
        options: list[str] | None = None,
    ) -> str: ...


class AutoApproveInteractor:
    """Scripted interactor that approves everything and answers option 1.

    Used by tests that exercise execution without interaction.
    """

    def confirm_plan(self, preview: dict[str, Any]) -> bool:
        return True

    def confirm_task(
        self, task_id: str, level: str, message: str, affected: list[str]
    ) -> bool:
        return True

    def ask(
        self,
        task_id: str,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        return options[0] if options else "yes"


class TerminalInteractor:
    """Prompt the user directly on the terminal.

    ``verbose`` renders the plan preview and questions with extra context.
    Reads from ``sys.stdin`` (fallback to ``input()``) so non-interactive
    sessions are handled by the caller (relay mode) instead.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose

    def _tty(self) -> bool:
        try:
            return bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            return False

    def confirm_plan(self, preview: dict[str, Any]) -> bool:
        if not self._tty():
            return False
        verbosity = 2 if self.verbose else 0
        print(format_plan_preview(preview, verbosity=verbosity))
        answer = self._read_choice(
            "Executar? (sim/nao) ", {"sim", "s", "yes", "y"}, False
        )
        return answer

    def confirm_task(
        self, task_id: str, level: str, message: str, affected: list[str]
    ) -> bool:
        if not self._tty():
            return False
        print(format_confirmation(task_id, level, message, affected))
        return self._read_choice(
            "Continuar? (sim/nao) ", {"sim", "s", "yes", "y"}, False
        )

    def ask(
        self,
        task_id: str,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        if not self._tty():
            return ""
        print(format_question(task_id, question, options))
        if options:
            print("Resposta (número ou texto):")
        return input("> ").strip()

    def _read_choice(self, prompt: str, accepted_yes: set[str], default: bool) -> bool:
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                return default
            if answer in accepted_yes:
                return True
            if answer in {"n", "nao", "não", "no"}:
                return False


#: Environment variable that forces non-interactive confirmation behavior.
def confirm_level_from_env(default: str = "HIGH") -> str:
    level = str(os.getenv("EXECUTOR_CONFIRM_LEVEL") or default).strip().upper()
    return level if level else default


__all__ = [
    "AutoApproveInteractor",
    "ExecutorInteractor",
    "TerminalInteractor",
    "confirm_level_from_env",
]
