"""Tests for the Executor Agent core: consuming a planner plan, executing
atomic tasks with isolated context, dependency handling, state machine,
budget limits and structured results."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.executor import (
    AcceptanceValidator,
    AutoApproveInteractor,
    ExecutorAgent,
    ExecutorBudget,
    TaskResult,
    TaskState,
    create_executor_tool,
)
from src.executor.state import InvalidTransition
from src.tools.registry import ToolRegistry, build_default_registry


class FakeProvider:
    """Scripted provider: returns the next queued response, then ``default``."""

    def __init__(self, responses=None, default=None):
        self.responses = list(responses or [])
        self.default = default or "tarefa concluída: arquivos criados."
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        if self.responses:
            return self.responses.pop(0)
        return self.default


def task(
    title,
    objective,
    files,
    context,
    changes,
    criteria,
    evidence,
    dependencies=None,
    status="READY_FOR_EXECUTION",
    **extra,
):
    base = {
        "title": title,
        "objective": objective,
        "status": status,
        "dependencies": dependencies or [],
        "files": files,
        "context": context,
        "expected_changes": changes,
        "acceptance_criteria": criteria,
        "evidence": evidence,
    }
    base.update(extra)
    return base


def plan(goal, tasks):
    return {"goal": goal, "summary": "resumo", "tasks": tasks}


def write_call(file_path, content):
    return json.dumps(
        {
            "tool": "write_file",
            "action": "write",
            "params": {"file_path": file_path, "content": content},
        }
    )


class ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def make_agent(self, provider, **kw):
        registry = build_default_registry()
        kw.setdefault("interactor", AutoApproveInteractor())
        return ExecutorAgent(provider, registry, root=str(self.root), **kw)

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


class TestTaskStateMachine(unittest.TestCase):
    def test_required_states_exist(self):
        for state in (
            "WAITING_CONFIRMATION",
            "READY",
            "EXECUTING",
            "VALIDATING",
            "COMPLETED",
            "FAILED",
            "BLOCKED",
            "CANCELLED",
        ):
            self.assertTrue(TaskState.is_valid(state), state)

    def test_future_states_are_represented(self):
        for state in (
            "NEED_DISCOVERY",
            "NEED_USER_INPUT",
            "NEED_CONFIRMATION",
            "REPAIRING",
            "REPLAN_REQUIRED",
        ):
            self.assertTrue(TaskState.is_valid(state), state)

    def test_active_transitions(self):
        self.assertEqual(
            TaskState.transition(TaskState.READY, TaskState.EXECUTING), "EXECUTING"
        )
        self.assertEqual(
            TaskState.transition(TaskState.EXECUTING, TaskState.VALIDATING),
            "VALIDATING",
        )
        self.assertEqual(
            TaskState.transition(TaskState.VALIDATING, TaskState.COMPLETED), "COMPLETED"
        )
        self.assertEqual(
            TaskState.transition(TaskState.VALIDATING, TaskState.FAILED), "FAILED"
        )
        self.assertEqual(
            TaskState.transition(TaskState.BLOCKED, TaskState.READY), "READY"
        )

    def test_illegal_transition_raises(self):
        with self.assertRaises(InvalidTransition):
            TaskState.transition(TaskState.COMPLETED, TaskState.READY)
        with self.assertRaises(InvalidTransition):
            TaskState.transition(TaskState.READY, TaskState.VALIDATING)
        with self.assertRaises(InvalidTransition):
            TaskState.transition(TaskState.CANCELLED, TaskState.READY)

    def test_dependency_helpers(self):
        self.assertTrue(TaskState.satisfies_dependency(TaskState.COMPLETED))
        self.assertFalse(TaskState.satisfies_dependency(TaskState.READY))
        for state in (
            TaskState.FAILED,
            TaskState.BLOCKED,
            TaskState.CANCELLED,
            TaskState.NEED_DISCOVERY,
            TaskState.REPLAN_REQUIRED,
        ):
            self.assertTrue(TaskState.blocks_dependents(state), state)


class TestSimpleTask(ExecutorTestCase):
    def test_single_task_completes(self):
        provider = FakeProvider(
            [
                write_call("src/hello.py", "print('hi')\n"),
                "Done: created src/hello.py",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar hello",
                [
                    task(
                        "criar modulo",
                        "criar src/hello.py",
                        ["src/hello.py"],
                        "arquivo novo",
                        ["novo arquivo hello.py"],
                        ["src/hello.py existe"],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["completed"], ["TASK-001"])
        row = result["tasks"][0]
        self.assertEqual(row["status"], "COMPLETED")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["task_id"], "TASK-001")
        self.assertEqual(res["status"], "COMPLETED")
        self.assertIn("src/hello.py", res["changed_files"])
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["output"], "Done: created src/hello.py")
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["attempts"], 1)
        self.assertTrue(res["timestamps"]["started_at"])
        self.assertTrue(res["timestamps"]["finished_at"])
        self.assertIsNotNone(res["validation_result"])
        checks = {c["check"]: c["ok"] for c in res["validation_result"]["checks"]}
        self.assertTrue(checks["tool_success"])
        self.assertTrue(checks["target_files_exist"])
        self.assertEqual(self.read("src/hello.py"), "print('hi')\n")
        self.assertEqual(result["metrics"]["tasks_executed"], 1)


class TestMultipleTasks(ExecutorTestCase):
    def test_two_independent_tasks_both_complete(self):
        provider = FakeProvider(
            [
                write_call("src/a.py", "a\n"),
                "A done",
                write_call("src/b.py", "b\n"),
                "B done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar a e b",
                [
                    task(
                        "tarefa A",
                        "criar a.py",
                        ["src/a.py"],
                        "ctx a",
                        ["a.py"],
                        ["a.py existe"],
                        "ev",
                    ),
                    task(
                        "tarefa B",
                        "criar b.py",
                        ["src/b.py"],
                        "ctx b",
                        ["b.py"],
                        ["b.py existe"],
                        "ev",
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(sorted(result["completed"]), ["TASK-001", "TASK-002"])
        self.assertEqual(result["metrics"]["tasks_executed"], 2)
        self.assertEqual(result["metrics"]["tool_calls"], 2)
        self.assertEqual(self.read("src/a.py"), "a\n")
        self.assertEqual(self.read("src/b.py"), "b\n")


class TestDependencies(ExecutorTestCase):
    def test_dependent_runs_after_dependency_completes(self):
        provider = FakeProvider(
            [
                write_call("src/base.py", "base\n"),
                "base done",
                write_call("src/derived.py", "derived\n"),
                "derived done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "base + derived",
                [
                    task(
                        "base",
                        "criar base.py",
                        ["src/base.py"],
                        "ctx base",
                        ["base.py"],
                        ["base.py existe"],
                        "ev CTX_BASE_UNIQUE",
                    ),
                    task(
                        "derived",
                        "criar derived.py",
                        ["src/derived.py"],
                        "usa base",
                        ["derived.py"],
                        ["derived.py existe"],
                        "ev",
                        dependencies=["TASK-001"],
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["completed"], ["TASK-001", "TASK-002"])
        # The dependency task's write must come first in the provider stream.
        self.assertIn("src/base.py", provider.calls[0][0])
        self.assertIn("TASK-002", provider.calls[2][0])
        # Isolation: task 1's unique context never leaks into task 2's prompt.
        self.assertNotIn("CTX_BASE_UNIQUE", provider.calls[2][0])


class TestIndependentAndBlocked(ExecutorTestCase):
    def test_failed_dependency_blocks_dependent_but_not_independents(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "run_command",
                        "action": "run",
                        "params": {"command": "exit 1"},
                    }
                ),
                "command failed",
                write_call("src/b.py", "b\n"),
                "B done",
                write_call("src/d.py", "d\n"),
                "D done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "cadeia",
                [
                    task(
                        "A",
                        "criar a.py",
                        ["src/a.py"],
                        "ctx a",
                        ["a.py"],
                        ["a.py existe"],
                        "ev",
                    ),
                    task(
                        "B",
                        "criar b.py",
                        ["src/b.py"],
                        "ctx b",
                        ["b.py"],
                        ["b.py existe"],
                        "ev",
                    ),
                    task(
                        "C",
                        "criar c.py",
                        ["src/c.py"],
                        "depende de A",
                        ["c.py"],
                        ["c.py existe"],
                        "ev",
                        dependencies=["TASK-001"],
                    ),
                    task(
                        "D",
                        "criar d.py",
                        ["src/d.py"],
                        "depende de B",
                        ["d.py"],
                        ["d.py existe"],
                        "ev",
                        dependencies=["TASK-002"],
                    ),
                ],
            ),
            # Repair is disabled so the scripted provider stream stays
            # deterministic (TASK-001's failure must not consume the queued
            # responses for TASK-002/TASK-004).
            max_retries=0,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["TASK-001"]["status"], "FAILED")
        self.assertEqual(result["results"]["TASK-002"]["status"], "COMPLETED")
        self.assertEqual(result["results"]["TASK-003"]["status"], "BLOCKED")
        self.assertIn("blocked by dependency", result["results"]["TASK-003"]["reason"])
        self.assertEqual(result["results"]["TASK-004"]["status"], "COMPLETED")
        # Independent tasks were not blocked by the failed one.
        self.assertIn("src/b.py", result["results"]["TASK-002"]["changed_files"])
        self.assertIn("src/d.py", result["results"]["TASK-004"]["changed_files"])
        # A BLOCKED task is never executed.
        self.assertEqual(result["results"]["TASK-003"]["tool_calls"], [])

    def test_plan_level_blocked_task_unblocks_when_dependency_completes(self):
        provider = FakeProvider(
            [
                write_call("src/a.py", "a\n"),
                "A done",
                write_call("src/b.py", "b\n"),
                "B done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "bloqueado no plano",
                [
                    task(
                        "A",
                        "criar a.py",
                        ["src/a.py"],
                        "ctx a",
                        ["a.py"],
                        ["a.py existe"],
                        "ev",
                    ),
                    task(
                        "B",
                        "criar b.py",
                        ["src/b.py"],
                        "espera A",
                        ["b.py"],
                        ["b.py existe"],
                        "ev",
                        dependencies=["TASK-001"],
                        status="BLOCKED",
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["completed"], ["TASK-001", "TASK-002"])
        self.assertEqual(result["results"]["TASK-002"]["status"], "COMPLETED")


class TestFailedTask(ExecutorTestCase):
    def test_validation_failure_marks_task_failed(self):
        # The model writes a file, but not the declared target: tool_success
        # passes while target_files_exist fails -> FAILED.
        provider = FakeProvider(
            [
                write_call("src/other.py", "x\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar alvo",
                [
                    task(
                        "alvo",
                        "criar src/x.py",
                        ["src/x.py"],
                        "arquivo novo",
                        ["x.py"],
                        ["x.py existe"],
                        "ev",
                    )
                ],
            )
        )
        self.assertEqual(result["status"], "partial")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("target_files_exist", res["reason"])
        self.assertFalse(res["validation_result"]["ok"])

    def test_no_tool_success_is_failed(self):
        # The model answers without doing any work: tool_success fails.
        provider = FakeProvider(["nada feito"])
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar alvo",
                [
                    task(
                        "alvo",
                        "criar src/x.py",
                        ["src/x.py"],
                        "novo",
                        ["x.py"],
                        ["x.py existe"],
                        "ev",
                    )
                ],
            )
        )
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("tool_success", res["reason"])


class TestPreservation(ExecutorTestCase):
    def test_completed_task_is_not_rerun(self):
        provider = FakeProvider(
            [
                write_call("src/b.py", "b\n"),
                "B done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "a pronto, b novo",
                [
                    task(
                        "A (pronto)",
                        "a.py pronto",
                        ["src/a.py"],
                        "já feito",
                        ["a.py"],
                        ["a.py existe"],
                        "ev",
                        status="COMPLETED",
                    ),
                    task(
                        "B",
                        "criar b.py",
                        ["src/b.py"],
                        "ctx b",
                        ["b.py"],
                        ["b.py existe"],
                        "ev",
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"]["TASK-001"]["status"], "COMPLETED")
        # The provider never saw the completed task: its first prompt is B.
        self.assertIn("TASK-002", provider.calls[0][0])
        self.assertNotIn("TASK-001", provider.calls[0][0])
        # Preserved tasks still get a lightweight result for uniform rendering.
        self.assertEqual(result["tasks"][0]["result"]["status"], "COMPLETED")


class TestLimits(ExecutorTestCase):
    def test_max_tasks_limits_execution(self):
        provider = FakeProvider(
            [
                write_call("src/a.py", "a\n"),
                "A done",
                write_call("src/b.py", "b\n"),
                "B done",
                write_call("src/c.py", "c\n"),
                "C done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "três arquivos",
                [
                    task(f"T{i}", f"criar {f}", [f], "ctx", [f], [f"existe {f}"], "ev")
                    for i, f in enumerate(["src/a.py", "src/b.py", "src/c.py"], 1)
                ],
            ),
            max_tasks=2,
        )
        self.assertEqual(result["status"], "partial")
        self.assertIn("max_tasks reached", result["reason"])
        self.assertEqual(result["metrics"]["tasks_executed"], 2)
        self.assertEqual(sorted(result["completed"]), ["TASK-001", "TASK-002"])
        self.assertEqual(result["skipped"], ["TASK-003"])
        self.assertIn("not executed", result["results"]["TASK-003"]["reason"])
        self.assertFalse((self.root / "src/c.py").exists())

    def test_max_tool_calls_fails_task(self):
        provider = FakeProvider(
            default=write_call("src/loop.py", "x\n"),
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "loop",
                [
                    task(
                        "loop",
                        "criar loop.py",
                        ["src/loop.py"],
                        "ctx",
                        ["loop.py"],
                        ["loop.py existe"],
                        "ev",
                    )
                ],
            ),
            max_tool_calls=2,
        )
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("max tool calls reached", res["reason"])
        self.assertEqual(len(res["tool_calls"]), 2)

    def test_max_context_tokens_fails_task(self):
        provider = FakeProvider([write_call("src/x.py", "x\n"), "done"])
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "ctx",
                [
                    task(
                        "ctx",
                        "criar x.py",
                        ["src/x.py"],
                        "ctx",
                        ["x.py"],
                        ["x.py existe"],
                        "ev",
                    )
                ],
            ),
            max_context_tokens=1,
        )
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("context exceeded", res["reason"])
        self.assertEqual(res["tool_calls"], [])


class TestIsolatedContext(ExecutorTestCase):
    def test_task_context_does_not_leak(self):
        provider = FakeProvider(
            [
                write_call("src/a.py", "a\n"),
                "A done",
                write_call("src/b.py", "b\n"),
                "B done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "isolamento",
                [
                    task(
                        "A",
                        "criar a.py",
                        ["src/a.py"],
                        "TOKEN_TASK_ONE contexto exclusivo da A",
                        ["a.py"],
                        ["a.py existe"],
                        "ev",
                    ),
                    task(
                        "B",
                        "criar b.py",
                        ["src/b.py"],
                        "TOKEN_TASK_TWO contexto exclusivo da B",
                        ["b.py"],
                        ["b.py existe"],
                        "ev",
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(len(provider.calls), 4)

        first_current, first_config = provider.calls[0]
        second_current, second_config = provider.calls[2]
        second_current_after, second_config_after = provider.calls[3]

        # Task A sees its own context and nothing from task B.
        self.assertIn("TOKEN_TASK_ONE", first_current)
        self.assertNotIn("TOKEN_TASK_TWO", first_current)
        self.assertNotIn("TOKEN_TASK_TWO", first_config)

        # Task B never sees task A's context, in neither the task prompt nor
        # the config/system prompt (fresh Memory per task).
        self.assertIn("TOKEN_TASK_TWO", second_current)
        self.assertNotIn("TOKEN_TASK_ONE", second_current)
        self.assertNotIn("TOKEN_TASK_ONE", second_config)
        self.assertNotIn("TOKEN_TASK_ONE", second_current_after)
        self.assertNotIn("TOKEN_TASK_ONE", second_config_after)


class TestInvalidPlan(ExecutorTestCase):
    def test_empty_tasks_is_error(self):
        agent = self.make_agent(FakeProvider([]))
        result = agent.execute({"goal": "x", "tasks": []})
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["errors"])

    def test_non_dict_plan_is_error(self):
        agent = self.make_agent(FakeProvider([]))
        result = agent.execute("not a plan")
        self.assertEqual(result["status"], "error")

    def test_dependency_cycle_is_error(self):
        provider = FakeProvider([])
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "ciclo",
                [
                    task(
                        "A",
                        "a",
                        ["src/a.py"],
                        "c",
                        ["e"],
                        ["a"],
                        "d",
                        dependencies=["TASK-002"],
                    ),
                    task(
                        "B",
                        "b",
                        ["src/b.py"],
                        "c",
                        ["e"],
                        ["b"],
                        "d",
                        dependencies=["TASK-001"],
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("cycle" in err for err in result["errors"]))


class TestNeedsDiscovery(ExecutorTestCase):
    def test_discovery_required_task_is_reported_not_executed(self):
        provider = FakeProvider(
            [
                write_call("src/r.py", "r\n"),
                "R done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "com discovery",
                [
                    task(
                        "D",
                        "descobrir e criar",
                        ["src/x.py"],
                        "",
                        [],
                        [],
                        "",
                        status="DISCOVERY_REQUIRED",
                    ),
                    task(
                        "R",
                        "criar r.py",
                        ["src/r.py"],
                        "ctx",
                        ["r.py"],
                        ["r.py existe"],
                        "ev",
                    ),
                ],
            )
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["needs_discovery"], ["TASK-001"])
        self.assertEqual(result["results"]["TASK-001"]["status"], "NEED_DISCOVERY")
        self.assertEqual(result["completed"], ["TASK-002"])


class TestToolSpecIntegration(ExecutorTestCase):
    def test_create_executor_tool_dispatch(self):
        provider = FakeProvider([write_call("src/x.py", "x\n"), "done"])
        registry = build_default_registry()
        spec = create_executor_tool(provider, registry, root=str(self.root))
        registry.register(spec.name, spec)

        self.assertIn("executor", registry.list_tools())
        out = registry.dispatch(
            "executor",
            "run",
            plan=plan(
                "via tool",
                [
                    task(
                        "x",
                        "criar x.py",
                        ["src/x.py"],
                        "ctx",
                        ["x.py"],
                        ["x.py existe"],
                        "ev",
                    )
                ],
            ),
            confirm=True,
        )
        self.assertEqual(out["status"], "success")
        status = registry.dispatch("executor", "status")
        self.assertEqual(status["status"], "last_run")
        self.assertEqual(registry.dispatch("executor", "pending")["status"], "last_run")
        self.assertEqual(registry.dispatch("executor", "abort")["status"], "aborted")


class TestResultStructure(unittest.TestCase):
    def test_task_result_to_dict_keys(self):
        result = TaskResult(task_id="TASK-001", status="COMPLETED")
        data = result.to_dict()
        for key in (
            "task_id",
            "status",
            "changed_files",
            "tool_calls",
            "output",
            "errors",
            "warnings",
            "attempts",
            "timestamps",
            "validation_result",
        ):
            self.assertIn(key, data)
        self.assertIn("started_at", data["timestamps"])

    def test_acceptance_validator_interface(self):
        root = tempfile.mkdtemp()
        validator = AcceptanceValidator(root=root)
        result = TaskResult(task_id="TASK-001")
        outcome = validator.validate({"files": ["missing.py"]}, result)
        self.assertFalse(outcome["ok"])
        self.assertTrue(any(c["check"] == "tool_success" for c in outcome["checks"]))
        self.assertTrue(
            any(c["check"] == "target_files_exist" for c in outcome["checks"])
        )


if __name__ == "__main__":
    unittest.main()
