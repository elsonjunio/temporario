"""Repair and recovery layer for the Executor (stage 3).

When acceptance validation fails, the Executor gives the model a bounded number
of repair attempts: the validation result is fed back as guidance and the task
re-executes (REPAIRING -> EXECUTING -> VALIDATING) without losing its history.
Only after ``max_retries`` is exhausted does the Executor classify the failure:

* ``FAILED`` — the task cannot meet its acceptance criteria (default);
* ``REPLAN_REQUIRED`` — the failure is structural (a declared target could not
  be created because it does not exist / cannot be found / is outside scope),
  so the planner must revise the plan first.

The classification is deterministic by default (``classify_failure``) and can
be swapped via the ``failure_classifier`` injection point.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from src.executor.result import TaskResult
from src.executor.state import TaskState
from src.planning.plan import as_list

#: Premise-consistent errors that suggest the plan's assumptions were wrong.
_REPLAN_ERROR_PATTERNS = (
    r"no such file",
    r"cannot find",
    r"not found",
    r"module not found",
    r"arquivo n[aã]o existe",
    r"does not exist",
    r"no such directory",
    r"cannot read",
    r"file not found",
)

#: Mutating tools whose failure on the declared target signals a wrong premise.
_TARGET_MUTATORS = ("write_file", "patch_file", "move_file")


def _premise_error(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _REPLAN_ERROR_PATTERNS)


def _targets_missing(validation_result: dict[str, Any]) -> bool:
    """True when the structural ``target_files_exist`` check failed."""
    for check in validation_result.get("checks", []):
        if check.get("check") == "target_files_exist":
            return not bool(check.get("ok"))
    return False


def classify_failure(
    task: dict[str, Any],
    result: TaskResult,
    validation_result: dict[str, Any],
) -> str:
    """Deterministic FAILED vs REPLAN_REQUIRED classification.

    REPLAN_REQUIRED when the declared targets are still missing after all
    repair attempts AND the recorded evidence shows the execution never managed
    to create them (a write/patch on a target failed, or the recorded errors
    match premise-consistent patterns such as "no such file"). Everything else
    is FAILED.
    """
    write_failures = [
        call
        for call in result.tool_calls
        if call.get("tool") in _TARGET_MUTATORS
        and isinstance(call.get("result"), dict)
        and call["result"].get("status") in ("failed", "error", "timeout")
    ]
    errors_text = " ".join(
        str(error.get("message") or error) for error in result.errors
    )
    premise = _premise_error(errors_text) or any(
        _premise_error(str(call["result"])) for call in write_failures
    )
    if _targets_missing(validation_result) and (write_failures or premise):
        return TaskState.REPLAN_REQUIRED
    return TaskState.FAILED


def build_repair_prompt(
    task: dict[str, Any],
    validation_result: dict[str, Any],
    attempt: int,
    max_retries: int,
) -> str:
    """Guidance prepended to the task context before a repair attempt."""
    failing = [
        entry.get("criterion")
        for entry in validation_result.get("acceptance_results", [])
        if entry.get("result") != "PASS"
    ]
    lines = [
        "VALIDATION FAILED - REPAIR",
        f"Repair attempt {attempt} of {max_retries}.",
        "Acceptance validation rejected your previous work. Address these "
        "failures with the tools; do not just repeat the same approach.",
    ]
    if failing:
        lines.append("Failed criteria:")
        lines.extend(f"- {criterion}" for criterion in failing)
    summary = validation_result.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
    commands = validation_result.get("commands") or []
    if commands:
        lines.append("Validation commands run:")
        lines.extend(f"- {c['command']} [{c['status']}]" for c in commands[:10])
    lines.append(
        "Make the required changes, then give your final plain-text summary "
        "(no JSON block)."
    )
    return "\n".join(lines)


def build_replanning_request(
    task: dict[str, Any],
    result: TaskResult,
    validation_result: dict[str, Any],
    problem: str,
    *,
    extra_evidence: list[str] | None = None,
    extra_files: list[str] | None = None,
    for_planner: str | None = None,
) -> dict[str, Any]:
    """Structured REPLAN_REQUIRED handoff for the planner.

    Mirrors the scope-growth handoff so the planner can merge it back via its
    ``replan`` action: task identity, the problem, the evidence that motivated
    it, what was already changed and which files are involved.
    """
    task_id = str(task.get("id") or "?")
    evidence: list[str] = []
    for call in result.discovery_calls:
        question = call.get("question")
        if question:
            evidence.append(f"discovery: {question}")
    evidence.extend(extra_evidence or [])
    failing = [
        entry
        for entry in validation_result.get("acceptance_results", [])
        if entry.get("result") != "PASS"
    ]
    if failing:
        evidence.append(
            "criteria não atendidas: "
            + "; ".join(
                f"{entry['criterion']} ({entry['result']})" for entry in failing[:10]
            )
        )
    for error in result.errors:
        message = str(error.get("message") or error)
        if message:
            evidence.append(f"erro: {message}")
    validation_summary = str(
        validation_result.get("summary") or "acceptance validation failed"
    )
    if validation_summary not in evidence:
        evidence.append(validation_summary)

    affected = sorted(
        {str(f) for f in as_list(task.get("files"))}
        | set(result.changed_files)
        | set(extra_files or [])
    )
    recommendation = for_planner or (
        f"Task {task_id} ({task.get('title') or ''}) failed acceptance "
        f"validation after retries and could not be completed as planned. "
        "Files already touched: "
        + (", ".join(sorted(result.changed_files)) or "none")
        + ". Planner should verify the task's assumptions (declared files, "
        "dependencies, scope) and revise the plan for this task and its "
        "dependents."
    )
    return {
        "task_id": task_id,
        "title": str(task.get("title") or ""),
        "status": TaskState.REPLAN_REQUIRED,
        "problem": problem,
        "evidence": evidence,
        "files": affected,
        "changes_made": sorted(result.changed_files),
        "reason": problem,
        "context": str(task.get("context") or ""),
        "recommendation": recommendation,
        "for_planner": recommendation,
    }


FailureClassifier = Callable[[dict[str, Any], TaskResult, dict[str, Any]], str]


__all__ = [
    "build_repair_prompt",
    "build_replanning_request",
    "classify_failure",
]
