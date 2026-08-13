from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory import Memory
from src.providers.opencode import OpenCodeProvider
from src.tools import registry as tool_registry
from src.tools.registry import ToolRegistry
from src.utils import (
    build_config_prompt,
    build_environment_info,
    classify_followup,
    parse_tool_call,
)

DEFAULT_MAX_ITERATIONS = 5


class Agent:
    """Loop that wires a provider, memory and a tool registry.

    The tool list travels inside the configuration prompt (no native function
    calling); tool calls are requested by the model as a JSON block that the
    loop parses and dispatches. The registry defaults to the base tools; extra
    tools (like the agentic orchestrator) are registered by the caller, keeping
    this core decoupled from them.
    """

    def __init__(
        self,
        provider: OpenCodeProvider,
        memory: Memory | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        registry: ToolRegistry | None = None,
        extra_tools: list[Any] | None = None,
        environment: str | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory if memory is not None else Memory()
        self.max_iterations = max_iterations
        self.registry = registry or tool_registry.default_registry
        for tool in extra_tools or []:
            self.registry.register(tool.name, tool)
        self.environment = environment or build_environment_info(
            workspace=str(Path.cwd())
        )
        self._pending_call: dict[str, Any] | None = None

    def run(
        self,
        user_prompt: str,
        *,
        max_iterations: int | None = None,
        **settings: Any,
    ) -> str:
        iterations = max_iterations or self.max_iterations
        current_prompt = user_prompt
        result: dict[str, Any] = {}
        self.memory.add_user(user_prompt)

        if self._pending_call is not None:
            verdict = classify_followup(user_prompt)
            if verdict == "approve":
                current_prompt = self._resume_pending()
            elif verdict == "abort":
                self._cancel_pending()
                return "The pending plan was cancelled. Nothing was executed."
            else:
                current_prompt = self._pending_guidance(user_prompt)

        for _ in range(iterations):
            config = build_config_prompt(
                tool_manuals=self.registry.get_manual(),
                memory=self.memory,
                environment=self.environment,
            )
            response = self.provider.infer(current_prompt, config, **settings)

            call = parse_tool_call(response)
            if call is None:
                self.memory.add_assistant(response)
                return response

            result = self.registry.dispatch(
                call["tool"],
                call["action"],
                **call["params"],
            )
            self.memory.add_tool(call["tool"], call["action"], call["params"], result)
            self.memory.add_assistant(response)

            if result.get("status") == "awaiting_confirmation":
                self._pending_call = {
                    "tool": call["tool"],
                    "preview": result,
                }
                pending_message = self._format_awaiting_message(result)
                self.memory.add_assistant(pending_message)
                return pending_message

            if result.get("status") in ("aborted", "failed"):
                self._pending_call = None

            current_prompt = (
                f"Tool result:\n{result}\n\n"
                "If that answers the request, give your final answer now "
                "(no JSON block). Otherwise keep working with the tools."
            )

        final = (
            f"I reached the maximum of {iterations} tool iterations without a "
            "final answer. Last tool result was:\n"
            f"{result}"
        )
        self.memory.add_assistant(final)
        return final

    @staticmethod
    def _format_awaiting_message(result: dict[str, Any]) -> str:
        """Render an ``awaiting_confirmation`` result as user-facing text.

        The summary is usually a list of plan steps; plain strings are kept
        as-is. The relay is deterministic so the plan always reaches the user
        without a fragile model round-trip.
        """
        summary = result.get("summary")
        if isinstance(summary, list):
            lines = []
            for step in summary:
                if not isinstance(step, dict):
                    continue
                tool = step.get("tool") or "?"
                action = step.get("action") or ""
                desc = step.get("description") or f"{tool} {action}".strip()
                lines.append(f"{step.get('step', '?')}. {desc}".strip())
                params = step.get("params")
                if params:
                    lines.append(f"   params: {params}")
            body = "\n".join(lines)
        else:
            body = str(summary or "")
        return (
            "Plano pronto para execução. Aguardando sua aprovação.\n\n"
            f"{body}\n\n"
            'Responda "sim" para aprovar, "não" para cancelar, ou '
            "descreva as mudanças desejadas."
        )

    def _resume_pending(self) -> str:
        """Resume the pending plan deterministically (no model round-trip)."""
        pending = self._pending_call
        assert pending is not None
        tool = pending["tool"]
        self._pending_call = None
        result = self.registry.dispatch(tool, "execute", confirm=True)
        self.memory.add_tool(tool, "execute", {"confirm": True}, result)
        if result.get("status") == "awaiting_confirmation":
            self._pending_call = pending
        return (
            f"Tool result:\n{result}\n\n"
            "If that answers the request, give your final answer now "
            "(no JSON block). Otherwise keep working with the tools."
        )

    def _cancel_pending(self) -> None:
        assert self._pending_call is not None
        self.registry.dispatch(self._pending_call["tool"], "abort")
        self._pending_call = None

    def _pending_guidance(self, user_prompt: str) -> str:
        """Send an ambiguous/conditional reply to the model with the pending
        plan as context, letting it decide between resume, re-plan or abort."""
        pending = self._pending_call
        assert pending is not None
        tool = pending["tool"]
        preview = pending["preview"]
        return (
            "A plan is awaiting confirmation. Pending plan:\n"
            f"{preview.get('summary')}\n\n"
            f'The user replied: "{user_prompt}".\n'
            "Decide how to proceed: if the user is approving, call "
            f"{tool} execute with confirm=true; if they changed the request, "
            "call run with the modified request; if they want to cancel, "
            "call abort."
        )
