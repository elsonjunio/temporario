from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory import Memory
from src.providers.opencode import OpenCodeProvider
from src.tools import registry as tool_registry
from src.tools.registry import ToolRegistry
from src.utils import build_config_prompt, build_environment_info, parse_tool_call

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
