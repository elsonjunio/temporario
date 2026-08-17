from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.executor.agent import EXECUTOR_INSTRUCTIONS, ExecutorAgent
from src.executor.budget import ExecutorBudget
from src.executor.interact import (
    AutoApproveInteractor,
    ExecutorInteractor,
    TerminalInteractor,
)
from src.executor.repair import (
    build_replanning_request,
    classify_failure,
)
from src.executor.result import TaskResult
from src.executor.state import TaskState
from src.executor.validate import (
    AcceptanceValidator,
    LLMCriterionJudge,
    ValidationManager,
)
from src.tools.base import ToolSpec
from src.tools.registry import ToolRegistry

EXECUTOR_MANUAL = (
    "executor: consumes a plan produced by the planner (or any plan with the "
    "same task schema) and executes its atomic tasks one at a time through the "
    "tool registry, respecting dependencies and per-task budgets.\n"
    "Actions:\n"
    "  - run (also execute)\n"
    "    Params: plan (dict, required): {goal, summary, tasks: [...]}. Each "
    "    task has id, title, objective, status (READY_FOR_EXECUTION | BLOCKED | "
    "    COMPLETED | DISCOVERY_REQUIRED), dependencies, files, context, "
    "    expected_changes, acceptance_criteria, evidence, risks, constraints.\n"
    "    confirm (bool, default false): approve the plan preview and start "
    "    executing without pausing. The first call returns an "
    "    'awaiting_confirmation' preview that must be relayed to the user "
    "    BEFORE any change; resume by calling execute again with confirm=true "
    "    (no plan needed) or cancel with abort.\n"
    "    Optional budget overrides: max_tasks, max_tool_calls, max_retries, "
    "    max_context_tokens, max_discovery_calls, max_discovery_context_chars, "
    "    max_validation_commands, validation_timeout, "
    "    max_browser_validation_steps.\n"
    "    Optional: confirm_level (LOW|MEDIUM|HIGH|CRITICAL|NEVER) raises or "
    "    lowers the risk gate for this run.\n"
    '    Returns: {"status": "success"|"partial"|"error", "goal", "summary", '
    '"tasks": [{task_id, title, status, reason, result}], "results": '
    '{task_id -> TaskResult}, "completed", "failed", "blocked", '
    '"needs_discovery", "replans", "cancelled", "skipped", "metrics", '
    '"errors", "warnings", "reason"}.\n'
    "    Each task runs in an isolated context (only the task + its own tool "
    "    results, discovery findings and user decisions, never the whole plan "
    "    or previous tasks) and moves through the executor state machine: "
    "READY -> EXECUTING -> VALIDATING -> COMPLETED | FAILED | BLOCKED.\n"
    "    A task starts only when all its dependencies are COMPLETED; a failed "
    "    dependency blocks its dependents while independent tasks continue. "
    "Tasks already COMPLETED/FAILED are preserved and never reset. The task is "
    "only COMPLETED after acceptance validation: structural checks (tool "
    "success + declared target files exist) plus per-criterion checks resolved "
    "by heuristics, targeted commands (compileall / npm run build / test "
    "runner), a functional browser pass and an LLM judge for unresolved "
    "criteria; each criterion ends PASS/FAIL/UNVERIFIED with evidence and "
    "UNVERIFIED counts as FAIL. A validation failure triggers a bounded repair "
    "loop (REPAIRING -> EXECUTING -> VALIDATING, up to max_retries=3 by "
    "default) feeding the failure back as guidance; after retries the task is "
    "classified FAILED, or REPLAN_REQUIRED when the failure is structural "
    "(declared target could not be created). The run result carries the "
    "structured extras: plan_status, completed_tasks, failed_tasks, "
    "blocked_tasks, cancelled_tasks, changed_files, validations, "
    "user_decisions, discovery_calls, replanning_requests and telemetry "
    "metrics (validation attempts, repairs, tokens, durations).\n"
    "  - status\n"
    "    Params: (none)\n"
    "    Returns: the structured result of the last run, a paused run, or "
    '"no_run".\n'
    "  - pending\n"
    "    Params: (none)\n"
    "    Returns what is awaiting action: a paused task (NEED_CONFIRMATION / "
    "NEED_USER_INPUT with the reason and, for confirmation, the exact "
    "operation), the pending plan preview, the last run, or 'no_pending'.\n"
    "  - answer\n"
    "    Params: task_id (required), answer (required, the user's reply).\n"
    "    Resumes a task paused on a user question (NEED_USER_INPUT), recording "
    "the decision and feeding it back into the task context.\n"
    "  - confirm_task\n"
    "    Params: task_id (required), confirm (bool, default true).\n"
    "    Approves or declines a task- or action-level risk confirmation "
    "(NEED_CONFIRMATION). Declining cancels the task.\n"
    "  - abort\n"
    "    Params: (none)\n"
    "    Cancels the pending plan or a paused run; nothing is executed when "
    "only a preview is pending.\n"
    "  - ask (model-side)\n"
    "    Params: question (required), options (optional list of choices).\n"
    "    Asks the user a preference mid-task; execution pauses (NEED_USER_INPUT) "
    "until the user answers via the answer action.\n"
    "  - replan_required (model-side)\n"
    "    Params: reason (required).\n"
    "    Stops the current task because discovery revealed its scope grew "
    "beyond the declared files (REPLAN_REQUIRED); the structured handoff "
    "(problem, evidence, files, changes_made, for_planner) is recorded for the "
    "planner and dependents are blocked.\n"
    "Safety: never marks a task COMPLETED because a tool returned success "
    "alone; never ignores dependencies; never executes a BLOCKED task; never "
    "resumes a FAILED task without an explicit state-machine decision; never "
    "modifies already-completed tasks. HIGH/CRITICAL tasks and destructive "
    "actions pause for explicit user confirmation unless confirm_level=NEVER "
    "or an inline interactor is installed."
)

_BUDGET_PARAMS = {
    "max_tasks",
    "max_tool_calls",
    "max_retries",
    "max_context_tokens",
    "max_discovery_calls",
    "max_discovery_context_chars",
    "max_validation_commands",
    "validation_timeout",
    "max_browser_validation_steps",
}


def create_executor_tool(
    provider: Any,
    registry: ToolRegistry,
    *,
    root: str = ".",
    name: str = "executor",
    budget: ExecutorBudget | None = None,
    validator: Any = None,
    environment: str | None = None,
    interactor: ExecutorInteractor | None = None,
    confirm_level: str | None = None,
    discovery_provider: Callable[[str], dict[str, Any]] | None = None,
    replanner: Callable[[dict[str, Any]], Any] | None = None,
    replan_extra_files_threshold: int = 1,
    failure_classifier: (
        Callable[[dict[str, Any], TaskResult, dict[str, Any]], str] | None
    ) = None,
) -> ToolSpec:
    """Build the special ``executor`` tool bound to a provider, registry and
    root.

    Unlike planner/discovery (read-only private registries), the executor uses
    the injected ``registry`` with its mutating tools, so it can actually write
    files and run commands.

    ``interactor`` is the inline confirmation channel: when set, the executor
    asks the human directly (plan approval, risk gates, questions) instead of
    pausing into NEED_* states. Leave it ``None`` in relay mode (the caller
    drives the pause via ``pending``/``answer``/``confirm_task``).
    ``discovery_provider(request)`` resolves the model's focused discovery
    questions; when ``None`` the executor falls back to the registered
    ``discovery`` tool. ``replanner`` receives the structured REPLAN_REQUIRED
    handoff; when ``None`` the run result alone carries it. ``validator``
    defaults to the full stage-3 ``ValidationManager`` (heuristics, targeted
    commands, browser pass, LLM judge). ``failure_classifier`` maps a task
    whose validation failed beyond ``max_retries`` to FAILED or
    REPLAN_REQUIRED (default: deterministic ``classify_failure``).

    Register the returned spec with ``ToolRegistry.register``; removing it
    keeps the agent working with the base tools - low coupling by design.
    """
    root_str = str(Path(root).resolve())
    agent = ExecutorAgent(
        provider,
        registry,
        root=root_str,
        budget=budget,
        environment=environment,
        validator=validator,
        interactor=interactor,
        confirm_level=confirm_level,
        discovery_provider=discovery_provider,
        replanner=replanner,
        replan_extra_files_threshold=replan_extra_files_threshold,
        failure_classifier=failure_classifier,
    )
    return ToolSpec(
        name=name,
        handlers={
            "run": agent.execute,
            "execute": agent.execute,
            "status": agent.status,
            "pending": agent.pending,
            "answer": agent.answer,
            "confirm_task": agent.confirm_task,
            "abort": agent.abort,
        },
        manual=EXECUTOR_MANUAL,
    )


__all__ = [
    "EXECUTOR_INSTRUCTIONS",
    "EXECUTOR_MANUAL",
    "AcceptanceValidator",
    "AutoApproveInteractor",
    "ExecutorAgent",
    "ExecutorBudget",
    "ExecutorInteractor",
    "LLMCriterionJudge",
    "TaskResult",
    "TaskState",
    "TerminalInteractor",
    "ValidationManager",
    "build_replanning_request",
    "classify_failure",
    "create_executor_tool",
]
