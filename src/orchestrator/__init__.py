from __future__ import annotations

from typing import Any

from src.memory import Memory
from src.orchestrator.orchestrator import MANUAL, Orchestrator
from src.tools.base import ToolSpec
from src.tools.registry import ToolRegistry


def create_orchestrator_tool(
    registry: ToolRegistry,
    provider: Any,
    memory: Memory | None = None,
    *,
    backup_root: str | None = None,
    max_plan_steps: int = 8,
    exec_retries: int = 1,
    root: str = ".",
    name: str = "orchestrator",
) -> ToolSpec:
    """Build the special orchestrator tool bound to a registry and provider.

    Registered via ``ToolRegistry.register(name, spec)``. If it is never
    registered (or unregistered later), the agent keeps working with the base
    tools — low coupling by design.
    """
    orch = Orchestrator(
        registry=registry,
        provider=provider,
        memory=memory,
        backup_root=backup_root,
        max_plan_steps=max_plan_steps,
        exec_retries=exec_retries,
        root=root,
    )
    return ToolSpec(
        name=name,
        handlers={
            "run": orch.run,
            "discover": orch.discover,
            "assess": orch.assess,
            "plan": orch.plan,
            "execute": orch.execute,
            "pending": orch.pending,
            "validate": orch.validate,
            "undo": orch.undo,
            "abort": orch.abort,
        },
        manual=MANUAL,
    )


__all__ = ["create_orchestrator_tool", "Orchestrator", "MANUAL"]
