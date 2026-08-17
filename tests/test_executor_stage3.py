"""Tests for the Executor stage-3 validation & recovery layer.

Covers the explicit VALIDATING stage, the bounded REPAIR loop, the
FAILED / REPLAN_REQUIRED classification, the layered ValidationManager
(heuristics, targeted commands, functional browser pass, LLM judge) and the
stage-3 telemetry / structured result fields.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.executor import (
    AutoApproveInteractor,
    ExecutorAgent,
    ExecutorBudget,
    LLMCriterionJudge,
    TaskResult,
    TaskState,
    build_replanning_request,
    classify_failure,
)
from src.executor.repair import build_repair_prompt
from src.executor.validate import _ValidationContext, _parse_verdicts
from src.tools.base import ToolSpec
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


def discovery_call(request):
    return json.dumps(
        {"tool": "discovery", "action": "run", "params": {"request": request}}
    )


def write_call(file_path, content):
    return json.dumps(
        {
            "tool": "write_file",
            "action": "write",
            "params": {"file_path": file_path, "content": content},
        }
    )


def patch_call(file_path, old, new):
    return json.dumps(
        {
            "tool": "patch_file",
            "action": "replace",
            "params": {"file_path": file_path, "old": old, "new": new},
        }
    )


def browser_call(action, **params):
    return json.dumps({"tool": "browser", "action": action, "params": params})


def fake_browser_spec():
    """A scriptable browser ToolSpec: snapshot body carries the token."""

    def open_(url=""):
        return {"status": "success", "url": url}

    def evaluate(expression=""):
        return {"status": "success", "result": [{"tag": "DIV", "text": "farewell"}]}

    def snapshot(**kwargs):
        return {"status": "success", "result": [{"tag": "BODY", "text": "farewell"}]}

    def network(**kwargs):
        return {"status": "success", "requests": []}

    return ToolSpec(
        name="browser",
        handlers={
            "open": open_,
            "evaluate": evaluate,
            "snapshot": snapshot,
            "network": network,
        },
        manual="navegador fake para testes",
    )


class ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def make_agent(self, provider, registry=None, **kw):
        registry = registry or build_default_registry()
        kw.setdefault("interactor", AutoApproveInteractor())
        return ExecutorAgent(provider, registry, root=str(self.root), **kw)

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation / repair integration
# ---------------------------------------------------------------------------


class TestStage3Validation(ExecutorTestCase):
    def test_validation_pass_completes_task(self):
        provider = FakeProvider(
            [
                write_call("src/app.py", "print('ok')\n"),
                "Done: created src/app.py",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["src/app.py existe"],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan_status"], "success")
        self.assertEqual(result["completed_tasks"], ["TASK-001"])
        self.assertEqual(result["failed_tasks"], [])
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")

        vr = res["validation_result"]
        self.assertEqual(vr["status"], "PASS")
        self.assertTrue(vr["ok"])
        self.assertIn("acceptance_results", vr)
        entry = vr["acceptance_results"][0]
        for key in ("criterion", "result", "evidence", "checked_by"):
            self.assertIn(key, entry)
        self.assertEqual(entry["result"], "PASS")
        self.assertEqual(entry["checked_by"], "heuristics")
        checks = {c["check"]: c["ok"] for c in vr["checks"]}
        self.assertTrue(checks["tool_success"])
        self.assertTrue(checks["target_files_exist"])

        self.assertEqual(res["attempts"], 1)
        self.assertEqual(res["repairs"], 0)
        self.assertEqual(res["validation_attempts"], 1)
        self.assertEqual(len(res["validation_history"]), 1)

        validation = result["validations"]["TASK-001"]
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["attempts"], 1)

        metrics = result["metrics"]
        self.assertEqual(metrics["tasks_completed_first_attempt"], 1)
        self.assertEqual(metrics["tasks_completed_after_validation"], 0)
        self.assertEqual(metrics["failures_caught_by_validation"], 0)
        self.assertEqual(metrics["retries_total"], 0)
        self.assertEqual(metrics["validation_attempts_total"], 1)
        self.assertEqual(metrics["tokens_used"], res["tokens_used"])

    def test_validation_failure_triggers_repair_and_passes(self):
        provider = FakeProvider(
            [
                write_call("src/wrong.py", "x\n"),
                "done primeira tentativa",
                write_call("src/app.py", "print('ok')\n"),
                "done reparo",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["src/app.py existe"],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["attempts"], 2)
        self.assertEqual(res["repairs"], 1)
        self.assertEqual(res["validation_attempts"], 2)
        self.assertEqual(len(res["validation_history"]), 2)
        self.assertEqual(res["validation_history"][0]["status"], "FAIL")
        self.assertEqual(res["validation_history"][1]["status"], "PASS")
        self.assertEqual(self.read("src/app.py"), "print('ok')\n")

        repair_prompt = provider.calls[2][0]
        self.assertIn("VALIDATION FAILED - REPAIR", repair_prompt)
        self.assertIn("Repair attempt 1 of 3", repair_prompt)

        metrics = result["metrics"]
        self.assertEqual(metrics["tasks_completed_after_validation"], 1)
        self.assertEqual(metrics["retries_total"], 1)
        self.assertEqual(metrics["validation_attempts_total"], 2)
        self.assertEqual(metrics["failures_caught_by_validation"], 0)
        self.assertEqual(result["validations"]["TASK-001"]["repairs"], 1)

    def test_repair_loop_respects_max_retries(self):
        provider = FakeProvider(
            [
                write_call("src/wrong.py", "x\n"),
                "done primeira tentativa",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["src/app.py existe"],
                        "greenfield",
                    )
                ],
            ),
            max_retries=2,
        )

        self.assertEqual(result["status"], "partial")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["attempts"], 3)
        self.assertEqual(res["repairs"], 2)
        self.assertEqual(res["validation_attempts"], 3)
        self.assertEqual(len(res["validation_history"]), 3)
        self.assertIn("target_files_exist", res["reason"])
        self.assertEqual(result["failed_tasks"], ["TASK-001"])

    def test_default_max_retries_is_three(self):
        self.assertEqual(ExecutorBudget().max_retries, 3)
        with patch.dict(os.environ, {"EXECUTOR_MAX_RETRIES": "7"}, clear=False):
            self.assertEqual(ExecutorBudget().with_env().max_retries, 7)

    def test_independent_task_completes_after_failure(self):
        provider = FakeProvider(
            [
                write_call("src/wrong.py", "x\n"),
                "a done (falhou)",
                write_call("src/b.py", "b\n"),
                "b done",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar a e b",
                [
                    task(
                        "criar a",
                        "criar src/a.py",
                        ["src/a.py"],
                        "arquivo novo",
                        ["novo arquivo a.py"],
                        ["src/a.py existe"],
                        "greenfield",
                    ),
                    task(
                        "criar b",
                        "criar src/b.py",
                        ["src/b.py"],
                        "arquivo novo",
                        ["novo arquivo b.py"],
                        ["src/b.py existe"],
                        "greenfield",
                    ),
                ],
            ),
            max_retries=0,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["completed"], ["TASK-002"])
        self.assertEqual(result["failed"], ["TASK-001"])
        self.assertEqual(result["completed_tasks"], ["TASK-002"])
        self.assertEqual(result["failed_tasks"], ["TASK-001"])
        self.assertEqual(result["results"]["TASK-001"]["status"], "FAILED")
        self.assertEqual(result["results"]["TASK-002"]["status"], "COMPLETED")
        self.assertEqual(result["metrics"]["tasks_executed"], 2)

    def test_repair_stops_on_tool_budget(self):
        provider = FakeProvider(default=write_call("src/wrong.py", "x\n"))
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["src/app.py existe"],
                        "greenfield",
                    )
                ],
            ),
            max_tool_calls=2,
        )

        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("max tool calls reached", res["reason"])
        self.assertEqual(res["repairs"], 0)
        self.assertEqual(res["attempts"], 1)


class TestExecutorDiscoveryMetric(ExecutorTestCase):
    """Planning-quality metric: how much discovery the Executor needed."""

    def _discovery_stub(self):
        def run(request):
            return {
                "status": "success",
                "report": {
                    "summary": f"resultado: {request}",
                    "relevant_files": ["src/app.py"],
                    "confidence": 0.9,
                },
            }

        return run

    def test_executor_discovery_total_and_per_task_questions(self):
        provider = FakeProvider(
            [
                discovery_call("qual arquivo cria o app?"),
                write_call("src/app.py", "print('ok')\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider, discovery_provider=self._discovery_stub())
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["src/app.py existe"],
                        "greenfield",
                    )
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["executor_discovery_total"], 1)
        validation = result["validations"]["TASK-001"]
        self.assertEqual(
            validation["discovery_calls"][0]["question"],
            "qual arquivo cria o app?",
        )


class TestReplanRequired(ExecutorTestCase):
    def test_replan_required_after_retries(self):
        provider = FakeProvider(
            [
                patch_call("src/x.py", "OLD", "NEW"),
                "tentativa um concluída",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar x",
                [
                    task(
                        "editar x",
                        "editar src/x.py",
                        ["src/x.py"],
                        "arquivo deve existir antes",
                        ["edicao de x.py"],
                        ["src/x.py existe"],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["replans"], ["TASK-001"])
        self.assertEqual(result["plan_status"], "partial")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "REPLAN_REQUIRED")
        self.assertEqual(res["repairs"], 3)
        self.assertTrue(res["replan"])

        req = res["replan"]
        for key in (
            "task_id",
            "title",
            "status",
            "problem",
            "evidence",
            "files",
            "changes_made",
            "reason",
            "context",
            "recommendation",
            "for_planner",
        ):
            self.assertIn(key, req)
        self.assertEqual(req["task_id"], "TASK-001")
        self.assertEqual(req["status"], "REPLAN_REQUIRED")
        self.assertIn("src/x.py", req["files"])
        self.assertTrue(req["problem"])

        replans = result["replanning_requests"]
        self.assertEqual(len(replans), 1)
        self.assertEqual(replans[0]["task_id"], "TASK-001")
        self.assertTrue(replans[0]["problem"])


# ---------------------------------------------------------------------------
# Layered ValidationManager
# ---------------------------------------------------------------------------


class TestTechnicalValidation(ExecutorTestCase):
    def test_compileall_technical_validation(self):
        provider = FakeProvider(
            [
                write_call("src/app.py", "print('ok')\n"),
                "done build",
            ]
        )
        agent = self.make_agent(provider)
        result = agent.execute(
            plan(
                "criar app",
                [
                    task(
                        "criar modulo",
                        "criar src/app.py",
                        ["src/app.py"],
                        "arquivo novo",
                        ["novo arquivo app.py"],
                        ["o build passa"],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")
        vr = res["validation_result"]
        self.assertEqual(vr["status"], "PASS")
        entry = vr["acceptance_results"][0]
        self.assertEqual(entry["result"], "PASS")
        self.assertEqual(entry["checked_by"], "command:build")
        commands = vr["commands"]
        self.assertTrue(any(c["kind"] == "build" for c in commands), commands)
        build_commands = [c for c in commands if c["kind"] == "build"]
        self.assertEqual(build_commands[0]["status"], "success")


class TestBrowserValidation(ExecutorTestCase):
    def test_browser_reuse_recorded_evidence(self):
        registry = build_default_registry()
        registry.register("browser", fake_browser_spec())
        (self.root / "src").mkdir()
        (self.root / "src" / "old.py").write_text("legacy\n", encoding="utf-8")
        provider = FakeProvider(
            [
                browser_call(
                    "evaluate", expression="document.querySelector('p').innerText"
                ),
                json.dumps(
                    {
                        "tool": "delete_file",
                        "action": "delete",
                        "params": {"path": "src/old.py"},
                    }
                ),
                "done remocao",
            ]
        )
        agent = self.make_agent(provider, registry=registry)
        result = agent.execute(
            plan(
                "remover old",
                [
                    task(
                        "remover arquivo",
                        "remover src/old.py e conferir a pagina",
                        ["src/old.py"],
                        "servidor local em http://localhost:5173",
                        ["arquivo removido"],
                        ['a página exibe "farewell"'],
                        "greenfield",
                    )
                ],
            )
        )

        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")
        vr = res["validation_result"]
        self.assertEqual(vr["status"], "PASS")
        entry = vr["acceptance_results"][0]
        self.assertEqual(entry["result"], "PASS")
        self.assertEqual(entry["checked_by"], "browser")
        self.assertFalse(
            any(c["kind"] == "browser" for c in vr["commands"]),
            "reuse must not launch the browser",
        )

    def test_browser_live_pass_bounded(self):
        registry = build_default_registry()
        registry.register("browser", fake_browser_spec())
        (self.root / "src").mkdir()
        (self.root / "src" / "old.py").write_text("legacy\n", encoding="utf-8")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "delete_file",
                        "action": "delete",
                        "params": {"path": "src/old.py"},
                    }
                ),
                "done remocao",
            ]
        )
        agent = self.make_agent(provider, registry=registry)
        result = agent.execute(
            plan(
                "remover old",
                [
                    task(
                        "remover arquivo",
                        "remover src/old.py e conferir a pagina",
                        ["src/old.py"],
                        "servidor local em http://localhost:5173",
                        ["arquivo removido"],
                        ['a página exibe "farewell"'],
                        "greenfield",
                    )
                ],
            ),
            max_browser_validation_steps=3,
        )

        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")
        vr = res["validation_result"]
        entry = vr["acceptance_results"][0]
        self.assertEqual(entry["result"], "PASS")
        self.assertEqual(entry["checked_by"], "browser")
        commands = vr["commands"]
        browser_commands = [c for c in commands if c["kind"] == "browser"]
        actions = [c["command"] for c in browser_commands]
        self.assertIn("browser:open", actions)
        self.assertIn("browser:snapshot", actions)
        self.assertLessEqual(len(browser_commands), 3)

    def test_browser_skipped_when_unavailable(self):
        provider = FakeProvider(
            [
                write_call("src/app.py", "print('ok')\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider)
        with patch.dict(os.environ, {"EXECUTOR_LLM_VALIDATION": "0"}, clear=False):
            result = agent.execute(
                plan(
                    "criar app",
                    [
                        task(
                            "criar modulo",
                            "criar src/app.py",
                            ["src/app.py"],
                            "arquivo novo",
                            ["novo arquivo app.py"],
                            ["a página renderiza corretamente"],
                            "greenfield",
                        )
                    ],
                ),
                max_retries=0,
            )

        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "FAILED")
        vr = res["validation_result"]
        self.assertEqual(vr["status"], "FAIL")
        entry = vr["acceptance_results"][0]
        self.assertEqual(entry["result"], "UNVERIFIED")
        self.assertIn("critérios UNVERIFIED contam como FAIL", vr["warnings"])


# ---------------------------------------------------------------------------
# Deterministic classification and handoff builders
# ---------------------------------------------------------------------------


def make_result(task_id="TASK-001"):
    result = TaskResult(task_id, "titulo", "objetivo")
    return result


def make_task(files=None):
    return {
        "id": "TASK-001",
        "title": "titulo",
        "objective": "objetivo",
        "files": files or ["src/x.py"],
        "context": "contexto",
        "acceptance_criteria": ["src/x.py existe"],
        "evidence": "ev",
    }


class TestClassifyFailure(unittest.TestCase):
    def test_failed_default_when_target_missing_but_no_write_failure(self):
        result = make_result()
        vr = {"checks": [{"check": "target_files_exist", "ok": False}]}
        self.assertEqual(classify_failure(make_task(), result, vr), TaskState.FAILED)

    def test_replan_when_target_missing_and_write_failed(self):
        result = make_result()
        result.tool_calls.append(
            {
                "tool": "write_file",
                "params": {"file_path": "src/x.py"},
                "result": {"status": "failed", "message": "file not found"},
            }
        )
        vr = {"checks": [{"check": "target_files_exist", "ok": False}]}
        self.assertEqual(
            classify_failure(make_task(), result, vr),
            TaskState.REPLAN_REQUIRED,
        )

    def test_replan_on_premise_error(self):
        result = make_result()
        result.errors = [{"message": "arquivo não existe no sistema"}]
        vr = {"checks": [{"check": "target_files_exist", "ok": False}]}
        self.assertEqual(
            classify_failure(make_task(), result, vr),
            TaskState.REPLAN_REQUIRED,
        )

    def test_failed_when_target_created_despite_write_failure(self):
        result = make_result()
        result.tool_calls.append(
            {
                "tool": "write_file",
                "params": {"file_path": "src/x.py"},
                "result": {"status": "failed", "message": "file not found"},
            }
        )
        result.changed_files = ["src/x.py"]
        vr = {"checks": [{"check": "target_files_exist", "ok": True}]}
        self.assertEqual(classify_failure(make_task(), result, vr), TaskState.FAILED)


class TestReplanningRequest(unittest.TestCase):
    def test_structure_and_files(self):
        result = make_result()
        result.changed_files = ["src/partial.py"]
        result.errors = [{"message": "no such file"}]
        vr = {
            "summary": "acceptance validation failed: src/x.py (FAIL)",
            "acceptance_results": [
                {
                    "criterion": "src/x.py existe",
                    "result": "FAIL",
                    "evidence": ["arquivos ausentes: src/x.py"],
                    "checked_by": "heuristics",
                }
            ],
        }
        req = build_replanning_request(
            make_task(["src/x.py"]),
            result,
            vr,
            "declared target could not be created",
            extra_files=["src/extra.py"],
            for_planner="revise o plano",
        )
        self.assertEqual(req["task_id"], "TASK-001")
        self.assertEqual(req["status"], "REPLAN_REQUIRED")
        self.assertEqual(req["problem"], "declared target could not be created")
        self.assertEqual(req["for_planner"], "revise o plano")
        self.assertIn("src/x.py", req["files"])
        self.assertIn("src/extra.py", req["files"])
        self.assertIn("src/partial.py", req["changes_made"])
        self.assertTrue(any("no such file" in e for e in req["evidence"]))
        self.assertTrue(any("src/x.py (FAIL)" in e for e in req["evidence"]))


class TestRepairPrompt(unittest.TestCase):
    def test_contents(self):
        prompt = build_repair_prompt(
            make_task(),
            {
                "summary": "acceptance validation failed: src/x.py (FAIL)",
                "acceptance_results": [
                    {
                        "criterion": "src/x.py existe",
                        "result": "FAIL",
                        "evidence": ["arquivos ausentes"],
                        "checked_by": "heuristics",
                    }
                ],
                "commands": [],
            },
            attempt=1,
            max_retries=3,
        )
        self.assertIn("VALIDATION FAILED - REPAIR", prompt)
        self.assertIn("Repair attempt 1 of 3", prompt)
        self.assertIn("src/x.py existe", prompt)


# ---------------------------------------------------------------------------
# LLM criterion judge
# ---------------------------------------------------------------------------


class TestLLMJudge(unittest.TestCase):
    def test_parse_verdicts_json(self):
        response = json.dumps(
            {
                "criteria": [
                    {
                        "criterion": "a cor #fff aplicada",
                        "result": "PASS",
                        "evidence": "estilo encontrado",
                        "reasoning": "x",
                    },
                    {
                        "criterion": "script carrega",
                        "result": "FAIL",
                        "evidence": "erro no console",
                        "reasoning": "y",
                    },
                ]
            }
        )
        verdicts = _parse_verdicts(response)
        self.assertEqual(verdicts["a cor #fff aplicada"].result, "PASS")
        self.assertEqual(verdicts["script carrega"].result, "FAIL")

    def test_parse_verdicts_bullets(self):
        response = (
            "PASS: a cor #fff aplicada — estilo encontrado\n" "FAIL - script carrega"
        )
        verdicts = _parse_verdicts(response)
        self.assertEqual(verdicts["a cor #fff aplicada"].result, "PASS")
        self.assertEqual(verdicts["script carrega"].result, "FAIL")

    def test_judge_applies_verdicts(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "criteria": [
                            {"criterion": "a cor #fff aplicada", "result": "PASS"},
                            {"criterion": "script carrega", "result": "FAIL"},
                        ]
                    }
                )
            ]
        )
        judge = LLMCriterionJudge(provider, root=".")
        entries = [
            {
                "criterion": "a cor #fff aplicada",
                "result": "UNVERIFIED",
                "evidence": [],
                "checked_by": "heuristics",
            },
            {
                "criterion": "script carrega",
                "result": "UNVERIFIED",
                "evidence": [],
                "checked_by": "heuristics",
            },
        ]
        result = make_result()
        result.tool_calls = [
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": "npm run build"},
                "result": {"status": "success", "returncode": 0},
            }
        ]
        task = make_task()
        task["acceptance_criteria"] = ["a cor #fff aplicada", "script carrega"]
        applied = judge.judge(
            _ValidationContext(".", ExecutorBudget()), task, result, entries
        )
        self.assertEqual(applied, 2)
        self.assertEqual(entries[0]["result"], "PASS")
        self.assertEqual(entries[0]["checked_by"], "llm")
        self.assertEqual(entries[1]["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
