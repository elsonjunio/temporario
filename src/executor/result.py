from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Cap for a single stored tool result inside ``TaskResult.tool_calls``, so a
#: long command output does not bloat the run result.
_STORAGE_RESULT_CHARS = 4096


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_now = now_iso


def _storage_result(result: Any) -> Any:
    """Cap a tool result dict for storage, keeping a preview when huge."""
    if not isinstance(result, dict):
        return result
    text = str(result)
    if len(text) <= _STORAGE_RESULT_CHARS:
        return result
    return {
        "truncated_for_history": True,
        "size": len(text),
        "preview": text[:_STORAGE_RESULT_CHARS],
    }


@dataclass
class TaskResult:
    """Structured outcome of a single task execution.

    ``changed_files`` lists the files a mutating tool reported touching during
    the task. ``tool_calls`` records every dispatched call (tool, action,
    params and the (capped) tool result). ``errors`` holds per-call failures
    and ``warnings`` non-fatal notes. ``validation_result`` is filled by the
    validator in the VALIDATING state and decides COMPLETED vs FAILED.
    ``discovery_calls`` records each focused discovery run (question + compact
    finding); ``user_inputs`` records questions answered by the user and
    ``confirmations`` the risk confirmations that were granted. ``replan``
    holds the REPLAN_REQUIRED structure when the task's scope grew or
    acceptance failed beyond retries. ``attempts`` counts execution attempts
    (1 + repairs); ``repairs`` the repair attempts and ``validation_history``
    every validation pass outcome (one per attempt), so failures caught by
    validation are auditable.
    """

    task_id: str
    status: str = ""
    changed_files: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1
    repairs: int = 0
    validation_attempts: int = 0
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    duration: float = 0.0
    validation_result: dict[str, Any] | None = None
    reason: str = ""
    discovery_calls: list[dict[str, Any]] = field(default_factory=list)
    user_inputs: list[dict[str, Any]] = field(default_factory=list)
    confirmations: list[dict[str, Any]] = field(default_factory=list)
    replan: dict[str, Any] | None = None

    def record_tool_call(
        self, tool: str, action: str, params: dict, result: dict
    ) -> None:
        self.tool_calls.append(
            {
                "tool": tool,
                "action": action,
                "params": params,
                "result": _storage_result(result),
            }
        )

    def record_error(self, tool: str, action: str, message: str) -> None:
        self.errors.append({"tool": tool, "action": action, "message": message})

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "changed_files": list(self.changed_files),
            "tool_calls": self.tool_calls,
            "output": self.output,
            "errors": self.errors,
            "warnings": self.warnings,
            "attempts": self.attempts,
            "repairs": self.repairs,
            "validation_attempts": self.validation_attempts,
            "validation_history": self.validation_history,
            "tokens_used": self.tokens_used,
            "timestamps": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration": self.duration,
            },
            "validation_result": self.validation_result,
            "reason": self.reason,
            "discovery_calls": self.discovery_calls,
            "user_inputs": self.user_inputs,
            "confirmations": self.confirmations,
            "replan": self.replan,
        }


def build_run_result(
    *,
    plan: dict[str, Any],
    task_order: list[str],
    final_states: dict[str, str],
    reasons: dict[str, str],
    results: dict[str, TaskResult],
    metrics: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    reason: str,
) -> dict[str, Any]:
    """Assemble the structured run result a presentation layer can render.

    ``status`` is ``success`` when every task ended COMPLETED, ``partial``
    otherwise (failed/blocked/needs-discovery/skipped tasks), ``error`` when
    the run never started (invalid plan / configuration).
    """
    tasks = plan.get("tasks") or []
    by_id = {str(task.get("id") or ""): task for task in tasks}

    rows: list[dict[str, Any]] = []
    completed: list[str] = []
    failed: list[str] = []
    blocked: list[str] = []
    needs_discovery: list[str] = []
    replans: list[str] = []
    cancelled: list[str] = []
    skipped: list[str] = []
    for task_id in task_order:
        state = final_states.get(task_id, "")
        row: dict[str, Any] = {
            "task_id": task_id,
            "title": str((by_id.get(task_id) or {}).get("title") or ""),
            "status": state,
            "reason": reasons.get(task_id, ""),
        }
        result = results.get(task_id)
        row["result"] = result.to_dict() if result is not None else None
        rows.append(row)
        if state == "COMPLETED":
            completed.append(task_id)
        elif state == "FAILED":
            failed.append(task_id)
        elif state == "BLOCKED":
            blocked.append(task_id)
        elif state == "NEED_DISCOVERY":
            needs_discovery.append(task_id)
        elif state == "REPLAN_REQUIRED":
            replans.append(task_id)
        elif state == "CANCELLED":
            cancelled.append(task_id)
        else:
            skipped.append(task_id)

    executed = int(metrics.get("tasks_executed") or 0)
    pending_attention = failed or blocked or needs_discovery or replans or cancelled
    status = (
        "success" if completed and not pending_attention and not skipped else "partial"
    )
    if not executed and errors:
        status = "error"

    # Stage 3 structured extras: per-task validation summaries, user decisions,
    # discovery/replanning records and the union of changed files.
    validations: dict[str, dict[str, Any]] = {}
    user_decisions: list[dict[str, Any]] = []
    discovery_calls: list[dict[str, Any]] = []
    replanning_requests: list[dict[str, Any]] = []
    changed_files: set[str] = set()
    for task_id in task_order:
        result = results.get(task_id)
        if result is None:
            continue
        validations[task_id] = {
            "status": (
                result.validation_result.get("status")
                if isinstance(result.validation_result, dict)
                else None
            ),
            "attempts": result.validation_attempts,
            "repairs": result.repairs,
            "history": [
                {"status": v.get("status"), "summary": v.get("summary")}
                for v in result.validation_history
            ],
            "tests": (
                result.validation_result.get("tests") or []
                if isinstance(result.validation_result, dict)
                else []
            ),
            "commands": (
                result.validation_result.get("commands") or []
                if isinstance(result.validation_result, dict)
                else []
            ),
            "discovery_calls": result.discovery_calls,
        }
        for decision in result.user_inputs:
            user_decisions.append({**decision, "task_id": task_id})
        for confirmation in result.confirmations:
            user_decisions.append({**confirmation, "task_id": task_id})
        for call in result.discovery_calls:
            discovery_calls.append({**call, "task_id": task_id})
        if result.replan:
            replanning_requests.append(result.replan)
        changed_files.update(result.changed_files)

    plan_status = "success" if completed and not pending_attention else "partial"
    if not executed and errors:
        plan_status = "error"

    return {
        "status": status,
        "goal": plan.get("goal", ""),
        "summary": plan.get("summary", ""),
        "tasks": rows,
        "results": {tid: r.to_dict() for tid, r in results.items()},
        "completed": completed,
        "failed": failed,
        "blocked": blocked,
        "needs_discovery": needs_discovery,
        "replans": replans,
        "cancelled": cancelled,
        "skipped": skipped,
        "plan_status": plan_status,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "blocked_tasks": blocked,
        "cancelled_tasks": cancelled,
        "changed_files": sorted(changed_files),
        "validations": validations,
        "user_decisions": user_decisions,
        "discovery_calls": discovery_calls,
        "replanning_requests": replanning_requests,
        "metrics": metrics,
        "errors": errors,
        "warnings": warnings,
        "reason": reason,
    }


__all__ = ["TaskResult", "build_run_result", "now_iso"]
