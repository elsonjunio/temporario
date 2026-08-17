from __future__ import annotations

from typing import Any, Callable

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
    discovery_fallback: Callable[..., dict[str, Any]] | None = None,
) -> ToolSpec:
    """Build the special orchestrator tool bound to a registry and provider.

    Registered via ``ToolRegistry.register(name, spec)``. If it is never
    registered (or unregistered later), the agent keeps working with the base
    tools — low coupling by design.

    ``discovery_fallback`` is an optional callable(request, terms, paths) that
    returns a discovery evidence dict. It is invoked automatically when the
    internal keyword discovery returns ``status != "ok"`` (balanced mode):
    the subagent's report fills the gap before planning. Returning poor
    evidence still aborts the flow.
    """
    orch = Orchestrator(
        registry=registry,
        provider=provider,
        memory=memory,
        backup_root=backup_root,
        max_plan_steps=max_plan_steps,
        exec_retries=exec_retries,
        root=root,
        discovery_fallback=discovery_fallback,
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
