from __future__ import annotations

from pathlib import Path
from typing import Any

from src.planning.agent import PLANNING_INSTRUCTIONS, PlanningAgent
from src.planning.budget import PlanningBudget
from src.planning.plan import (
    READY_STATUSES,
    VALID_STATUSES,
    normalize_task,
    validate_plan,
)
from src.tools.base import ToolSpec
from src.tools.registry import ToolRegistry

PLANNER_MANUAL = (
    "planner: Planning Agent. Transforms a user request into a plan of atomic, "
    "executable tasks for a separate Executor Agent. It runs read-only "
    "analysis and focused Discovery investigations on demand; it NEVER "
    "modifies files, runs commands, installs dependencies or creates commits.\n"
    "Actions:\n"
    "  - run\n"
    "    Params: request (str, required): what the user wants to achieve.\n"
    "      Optional budget overrides: max_iterations, max_discovery_calls,\n"
    "      max_tasks, max_context_tokens.\n"
    '    Returns: {"status": "success"|"partial", "request", '
    '"workspace", "plan": {"goal", "summary", "tasks", "issues", '
    '"warnings"}, "metrics"}. Each task has id, title, objective, status\n'
    "      (DISCOVERY_REQUIRED | READY_FOR_EXECUTION | BLOCKED | COMPLETED),\n"
    "      dependencies (task ids), files, context (task-specific only),\n"
    "      expected_changes, acceptance_criteria and evidence. Status\n"
    "      'partial' means some tasks are DISCOVERY_REQUIRED.\n"
    "  - plan: alias of run.\n"
    "  - discover\n"
    "    Params: question (str, required): ONE focused investigation question;\n"
    "      scope (str, optional path); optional budget overrides.\n"
    "    Returns: the discovery evidence for that question (delegates to the\n"
    "      discovery subagent).\n"
    "  - replan\n"
    "    Params: request (str, required); previous_plan (dict, optional);\n"
    "      task_results (list or dict, optional) mapping task id ->\n"
    '      "success"/"failed"/... or {"status": ..., "notes": ...}.\n'
    "    Returns: an updated plan; completed tasks stay COMPLETED and new or\n"
    "      revised tasks are added.\n"
    "  - pending\n"
    "    Params: (none)\n"
    '    Returns: the last produced plan, or "no_plan".\n'
    "Strategy: a task is READY_FOR_EXECUTION only with objective, files,\n"
    "context, expected_changes, acceptance_criteria and evidence defined;\n"
    "otherwise it is marked DISCOVERY_REQUIRED (never invented). A focused\n"
    "discovery call beats a guessed task."
)

_BUDGET_PARAMS = {
    "max_iterations",
    "max_discovery_calls",
    "max_tasks",
    "max_context_tokens",
    "max_context_chars",
}


def create_planning_tool(
    provider: Any,
    registry: ToolRegistry | None = None,
    *,
    root: str = ".",
    name: str = "planner",
    budget: PlanningBudget | None = None,
    discovery: Any = None,
    environment: str | None = None,
) -> ToolSpec:
    """Build the special ``planner`` tool bound to a provider and root.

    ``discovery`` may be a discovery ToolSpec/module or callable; when omitted
    the planner reuses the ``discovery`` entry of ``registry`` (if present) or
    builds its own ephemeral discovery agent. Register the returned spec with
    ``ToolRegistry.register``; removing it keeps the agent working with the
    base tools — low coupling by design.
    """
    root_str = str(Path(root).resolve())
    agent = PlanningAgent(
        provider,
        root=root_str,
        budget=budget,
        discovery=discovery,
        outer_registry=registry,
        environment=environment,
    )
    return ToolSpec(
        name=name,
        handlers={
            "run": agent.run,
            "plan": agent.run,
            "discover": agent.discover,
            "replan": agent.replan,
            "pending": agent.pending,
        },
        manual=PLANNER_MANUAL,
    )


__all__ = [
    "PLANNER_MANUAL",
    "PLANNING_INSTRUCTIONS",
    "PlanningAgent",
    "PlanningBudget",
    "READY_STATUSES",
    "VALID_STATUSES",
    "create_planning_tool",
    "normalize_task",
    "validate_plan",
]
