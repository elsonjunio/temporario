from __future__ import annotations

from typing import Iterable


class InvalidTransition(RuntimeError):
    """Raised when a task state transition is not allowed by the machine."""


class TaskState:
    """Single, consistent representation of a task's lifecycle state.

    Every state the Executor can enter is a constant here, so states are never
    ad-hoc strings scattered across the code. The machine is defined by
    ``_TRANSITIONS`` and enforced by :meth:`transition`.

    Active execution states: READY, EXECUTING, VALIDATING, COMPLETED, FAILED,
    BLOCKED, CANCELLED.

    Interaction states (stage 2): WAITING_CONFIRMATION (plan approval gate),
    NEED_DISCOVERY (investigation before/while executing), NEED_USER_INPUT /
    NEED_CONFIRMATION (interactive clarification / critical-action approval)
    and REPLAN_REQUIRED (scope grew or acceptance failed; hand back to the
    planner). A task pauses into one of the NEED_* states and resumes back into
    EXECUTING without losing its session context.

    Stage 3 (validation & recovery): REPAIRING is the bounded retry loop after
    a validation failure (VALIDATING -> FAILED -> REPAIRING -> EXECUTING ->
    VALIDATING); the task is only classified FAILED or REPLAN_REQUIRED once
    ``max_retries`` is exhausted.
    """

    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    READY = "READY"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"

    #: Interaction states used by stage 2 (pause and resume mid-task).
    NEED_DISCOVERY = "NEED_DISCOVERY"
    NEED_USER_INPUT = "NEED_USER_INPUT"
    NEED_CONFIRMATION = "NEED_CONFIRMATION"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"

    #: Bounded retry loop entered after a validation failure (stage 3).
    REPAIRING = "REPAIRING"

    #: States a task never leaves automatically.
    TERMINAL = frozenset({COMPLETED, FAILED, BLOCKED, CANCELLED})

    #: States the Executor enters for a task while its work is not finished.
    ACTIVE = frozenset({READY, EXECUTING, VALIDATING, REPAIRING})

    #: Pause states a task can be suspended in and later resumed from.
    PAUSED = frozenset({NEED_DISCOVERY, NEED_USER_INPUT, NEED_CONFIRMATION})

    #: States reserved for later stages; not entered yet.
    RESERVED = frozenset({WAITING_CONFIRMATION})

    #: Allowed transitions. A state missing here is terminal (or reserved) and
    #: only leaves through the explicit entries below.
    _TRANSITIONS: dict[str, frozenset[str]] = {
        WAITING_CONFIRMATION: frozenset({READY, CANCELLED}),
        READY: frozenset({EXECUTING, BLOCKED, CANCELLED, NEED_DISCOVERY}),
        EXECUTING: frozenset(
            {
                VALIDATING,
                FAILED,
                CANCELLED,
                BLOCKED,
                NEED_DISCOVERY,
                NEED_USER_INPUT,
                NEED_CONFIRMATION,
                REPLAN_REQUIRED,
            }
        ),
        VALIDATING: frozenset({COMPLETED, FAILED, BLOCKED}),
        BLOCKED: frozenset({READY, CANCELLED, REPLAN_REQUIRED}),
        COMPLETED: frozenset({REPLAN_REQUIRED}),
        FAILED: frozenset(
            {REPAIRING, NEED_DISCOVERY, NEED_USER_INPUT, REPLAN_REQUIRED}
        ),
        CANCELLED: frozenset(),
        NEED_DISCOVERY: frozenset({READY, EXECUTING, CANCELLED}),
        NEED_USER_INPUT: frozenset({READY, EXECUTING, CANCELLED}),
        NEED_CONFIRMATION: frozenset({READY, EXECUTING, CANCELLED, REPLAN_REQUIRED}),
        REPAIRING: frozenset({EXECUTING, VALIDATING, FAILED, CANCELLED}),
        REPLAN_REQUIRED: frozenset({READY, CANCELLED}),
    }

    def __init__(self) -> None:
        raise TypeError("TaskState is a namespace of constants; use TaskState.READY")

    @classmethod
    def all(cls) -> list[str]:
        """Every known state, active and reserved."""
        return list(cls._TRANSITIONS)

    @classmethod
    def is_valid(cls, state: str) -> bool:
        return state in cls._TRANSITIONS

    @classmethod
    def can_transition(cls, source: str, target: str) -> bool:
        if not cls.is_valid(source) or not cls.is_valid(target):
            return False
        return target in cls._TRANSITIONS[source]

    @classmethod
    def transition(cls, source: str, target: str) -> str:
        """Return ``target`` when the transition is legal, else raise
        ``InvalidTransition``."""
        if not cls.is_valid(source) or not cls.is_valid(target):
            raise InvalidTransition(f"unknown state: {source!r} -> {target!r}")
        if target not in cls._TRANSITIONS[source]:
            raise InvalidTransition(
                f"illegal task state transition: {source} -> {target}"
            )
        return target

    @classmethod
    def satisfies_dependency(cls, state: str) -> bool:
        """A task counts as a satisfied dependency for its dependents."""
        return state == cls.COMPLETED

    @classmethod
    def blocks_dependents(cls, state: str) -> bool:
        """A task in one of these states blocks its dependents."""
        return state in (
            cls.FAILED,
            cls.BLOCKED,
            cls.CANCELLED,
            cls.NEED_DISCOVERY,
            cls.REPLAN_REQUIRED,
        )


#: Mapping from Planning Agent task statuses to the Executor's initial state.
PLAN_STATUS_TO_STATE = {
    "READY_FOR_EXECUTION": TaskState.READY,
    "BLOCKED": TaskState.BLOCKED,
    "COMPLETED": TaskState.COMPLETED,
    "DISCOVERY_REQUIRED": TaskState.NEED_DISCOVERY,
}


def initial_state_from_plan_status(status: str) -> str:
    """Derive the Executor's initial task state from a planner task status."""
    state = PLAN_STATUS_TO_STATE.get(status or "")
    if state is None:
        raise ValueError(f"unmapped plan status {status!r}")
    return state


def plan_statuses_summary(states: Iterable[str]) -> dict[str, int]:
    """Count occurrences of each state, for metrics."""
    summary: dict[str, int] = {}
    for state in states:
        summary[state] = summary.get(state, 0) + 1
    return summary


__all__ = [
    "InvalidTransition",
    "PLAN_STATUS_TO_STATE",
    "TaskState",
    "initial_state_from_plan_status",
    "plan_statuses_summary",
]
