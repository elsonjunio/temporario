"""Stage 2 tests: interaction & control layer of the Executor Agent.

Covers plan confirmation (relay + inline), risk gates (task-level and
critical-action), user questions, focused discovery with budget/compaction,
REPLAN_REQUIRED on scope growth, resume without losing the session, abort,
state-machine transitions and the presenter layer.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.executor import (
    AcceptanceValidator,
    AutoApproveInteractor,
    ExecutorAgent,
    ExecutorBudget,
    TaskState,
    create_executor_tool,
)
from src.executor.presenter import (
    format_confirmation,
    format_plan_preview,
    format_question,
    format_result,
    summarize_risks,
)
from src.executor.result import TaskResult, build_run_result
from src.executor.risk import RiskLevel, classify_action, classify_task
from src.executor.state import InvalidTransition
from src.tools.registry import build_default_registry


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


class DecliningInteractor:
    """Interactor that rejects plan approval and risk confirmations."""

    def confirm_plan(self, preview):
        return False

    def confirm_task(self, task_id, level, message, affected):
        return False

    def ask(self, task_id, question, options=None):
        return ""


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


def delete_call(path):
    return json.dumps(
        {"tool": "delete_file", "action": "delete", "params": {"path": path}}
    )


def ask_call(question, options=None):
    params = {"question": question}
    if options:
        params["options"] = options
    return json.dumps({"tool": "executor", "action": "ask", "params": params})


def discovery_call(request):
    return json.dumps(
        {"tool": "discovery", "action": "run", "params": {"request": request}}
    )


class Stage2ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def make_agent(self, provider, **kw):
        registry = build_default_registry()
        kw.setdefault("interactor", None)
        return ExecutorAgent(provider, registry, root=str(self.root), **kw)

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


class TestPlanConfirmation(Stage2ExecutorTestCase):
    def test_preview_then_confirm_executes(self):
        provider = FakeProvider([write_call("src/x.py", "x\n"), "done"])
        agent = self.make_agent(provider)
        p = plan(
            "via preview",
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
        )

        first = agent.execute(p)
        self.assertEqual(first["status"], "awaiting_confirmation")
        self.assertEqual(first["task_count"], 1)
        self.assertFalse((self.root / "src/x.py").exists())
        self.assertEqual(provider.calls, [])

        pending = agent.pending()
        self.assertEqual(pending["status"], "pending_plan")

        second = agent.execute(confirm=True)
        self.assertEqual(second["status"], "success")
        self.assertEqual(self.read("src/x.py"), "x\n")

    def test_inline_interactor_skips_preview_pause(self):
        provider = FakeProvider([write_call("src/x.py", "x\n"), "done"])
        agent = self.make_agent(provider, interactor=AutoApproveInteractor())
        result = agent.execute(
            plan(
                "inline",
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
            )
        )
        self.assertEqual(result["status"], "success")

    def test_declined_plan_runs_nothing(self):
        provider = FakeProvider([write_call("src/x.py", "x\n"), "done"])
        agent = self.make_agent(provider, interactor=DecliningInteractor())
        result = agent.execute(
            plan(
                "recusado",
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
            )
        )
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse((self.root / "src/x.py").exists())
        self.assertEqual(provider.calls, [])

    def test_execute_without_plan_errors(self):
        agent = self.make_agent(FakeProvider([]))
        result = agent.execute()
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("no plan" in err for err in result["errors"]))


class TestRiskGates(Stage2ExecutorTestCase):
    def _high_task(self):
        return task(
            "migrar schema",
            "aplicar a migração do schema no banco",
            ["src/db.py"],
            "ctx migração",
            ["db.py alterado"],
            ["db.py existe"],
            "ev",
        )

    def test_task_level_confirmation_pauses_then_resumes(self):
        provider = FakeProvider([write_call("src/db.py", "db\n"), "done"])
        agent = self.make_agent(provider)
        self.assertEqual(classify_task(self._high_task()), RiskLevel.HIGH)

        result = agent.execute(plan("migrar", [self._high_task()]))
        self.assertEqual(result["status"], "awaiting_confirmation")

        result = agent.execute(confirm=True)
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(result["kind"], "confirmation")
        self.assertEqual(agent.status()["status"], "paused")

        resumed = agent.confirm_task("TASK-001", True)
        self.assertEqual(resumed["status"], "success")
        res = resumed["results"]["TASK-001"]
        self.assertEqual(res["status"], "COMPLETED")
        self.assertTrue(any(c["scope"] == "task" for c in res["confirmations"]))

    def test_task_level_decline_cancels_task(self):
        provider = FakeProvider([write_call("src/db.py", "db\n"), "done"])
        agent = self.make_agent(provider)
        agent.execute(plan("migrar", [self._high_task()]))
        agent.execute(confirm=True)

        result = agent.confirm_task("TASK-001", False)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["TASK-001"]["status"], "CANCELLED")
        self.assertEqual(result["cancelled"], ["TASK-001"])
        self.assertFalse((self.root / "src/db.py").exists())

    def test_action_level_confirmation_dispatches_after_approval(self):
        artifact = self.root / "tmp" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("lixo\n", encoding="utf-8")

        provider = FakeProvider([delete_call("tmp/artifact.txt"), "done"])
        agent = self.make_agent(provider)
        plan_obj = plan(
            "limpeza",
            [
                task(
                    "limpar",
                    "limpar arquivos temporários da pasta tmp",
                    ["tmp/artifact.txt"],
                    "ctx",
                    ["limpar tmp"],
                    ["arquivo não existe mais"],
                    "ev",
                )
            ],
        )
        self.assertEqual(classify_task(plan_obj["tasks"][0]), RiskLevel.LOW)

        agent.execute(plan_obj)
        result = agent.execute(confirm=True)
        # The LOW task has no task-level gate; the critical delete pauses.
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(agent.pending()["kind"], "confirmation")
        # The pending confirmation carries the exact action.
        pending = agent.pending()["pause"]
        self.assertIsNotNone(pending["pending"])
        self.assertEqual(pending["pending"]["tool"], "delete_file")

        result = agent.confirm_task("TASK-001", True)
        self.assertEqual(result["results"]["TASK-001"]["status"], "COMPLETED")
        self.assertFalse(artifact.exists())
        res = result["results"]["TASK-001"]
        self.assertTrue(any(c["scope"] == "action" for c in res["confirmations"]))

    def test_action_level_decline_keeps_file(self):
        artifact = self.root / "tmp" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("lixo\n", encoding="utf-8")

        provider = FakeProvider([delete_call("tmp/artifact.txt"), "done"])
        agent = self.make_agent(provider)
        agent.execute(
            plan(
                "limpeza",
                [
                    task(
                        "limpar",
                        "limpar arquivos temporários da pasta tmp",
                        ["tmp/artifact.txt"],
                        "ctx",
                        ["limpar tmp"],
                        ["arquivo não existe mais"],
                        "ev",
                    )
                ],
            )
        )
        agent.execute(confirm=True)
        result = agent.confirm_task("TASK-001", False)
        self.assertEqual(result["results"]["TASK-001"]["status"], "CANCELLED")
        self.assertTrue(artifact.exists())

    def test_confirm_level_never_disables_gate(self):
        artifact = self.root / "tmp" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("lixo\n", encoding="utf-8")

        provider = FakeProvider([delete_call("tmp/artifact.txt"), "done"])
        agent = self.make_agent(provider, confirm_level=RiskLevel.NEVER)
        result = agent.execute(
            plan(
                "limpeza",
                [
                    task(
                        "limpar",
                        "remover artefato",
                        ["tmp/artifact.txt"],
                        "ctx",
                        ["remover"],
                        ["removido"],
                        "ev",
                    )
                ],
            ),
            confirm=True,
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(artifact.exists())

    def test_confirm_level_env_override(self):
        with mock.patch.dict(
            os.environ, {"EXECUTOR_CONFIRM_LEVEL": "NEVER"}, clear=False
        ):
            agent = self.make_agent(FakeProvider([]))
            self.assertEqual(agent.confirm_level, RiskLevel.NEVER)

    def test_classify_action_levels(self):
        self.assertEqual(
            classify_action("delete_file", "delete", {"path": "a.txt"}),
            RiskLevel.CRITICAL,
        )
        self.assertEqual(
            classify_action("run_command", "run", {"command": "rm -rf /tmp/x"}),
            RiskLevel.CRITICAL,
        )
        self.assertEqual(
            classify_action("run_command", "run", {"command": "pip install requests"}),
            RiskLevel.HIGH,
        )
        self.assertEqual(
            classify_action("write_file", "write", {"file_path": "src/a.py"}),
            RiskLevel.MEDIUM,
        )
        self.assertEqual(
            classify_action("read_file", "read", {"file_path": "src/a.py"}),
            RiskLevel.LOW,
        )


class TestUserQuestions(Stage2ExecutorTestCase):
    def test_answer_injects_decision_and_resumes(self):
        provider = FakeProvider(
            [
                ask_call("qual banco usar?", ["sqlite", "postgres"]),
                write_call("src/db.py", "db\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider)
        agent.execute(
            plan(
                "db",
                [
                    task(
                        "db",
                        "escolher o nome do arquivo",
                        ["src/db.py"],
                        "ctx",
                        ["db.py"],
                        ["db.py existe"],
                        "ev",
                    )
                ],
            )
        )
        result = agent.execute(confirm=True)
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(result["kind"], "question")

        resumed = agent.answer("TASK-001", "usar postgres")
        self.assertEqual(resumed["status"], "success")
        res = resumed["results"]["TASK-001"]
        self.assertEqual(len(res["user_inputs"]), 1)
        self.assertEqual(res["user_inputs"][0]["answer"], "usar postgres")
        # The decision was fed back into the model's prompt after resuming.
        joined = "\n".join(p for p, _ in provider.calls)
        self.assertIn("usar postgres", joined)

    def test_inline_interactor_answers_automatically(self):
        provider = FakeProvider(
            [
                ask_call("qual banco?", ["sqlite", "postgres"]),
                write_call("src/db.py", "db\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider, interactor=AutoApproveInteractor())
        result = agent.execute(
            plan(
                "db",
                [
                    task(
                        "db",
                        "escolher o nome do arquivo",
                        ["src/db.py"],
                        "ctx",
                        ["db.py"],
                        ["db.py existe"],
                        "ev",
                    )
                ],
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["results"]["TASK-001"]["user_inputs"][0]["answer"], "sqlite"
        )

    def test_execute_confirm_true_on_question_pauses_errors(self):
        provider = FakeProvider(
            [ask_call("dúvida?"), write_call("src/a.py", "a\n"), "done"]
        )
        agent = self.make_agent(provider)
        agent.execute(
            plan(
                "q", [task("q", "criar a", ["src/a.py"], "c", ["a"], ["a existe"], "e")]
            )
        )
        agent.execute(confirm=True)
        result = agent.execute(confirm=True)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("answer" in err for err in result["errors"]))


class TestDiscovery(Stage2ExecutorTestCase):
    def _stub(self, extra_files=()):
        def run(request):
            files = ["src/x.py"] + list(extra_files)
            return {
                "status": "success",
                "report": {
                    "summary": f"resultado para: {request}",
                    "relevant_files": files,
                    "confidence": 0.9,
                },
            }

        return run

    def test_discovery_intercepted_and_compacted(self):
        provider = FakeProvider(
            [
                discovery_call("quem usa a classe X?"),
                write_call("src/x.py", "x\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider, discovery_provider=self._stub())
        result = agent.execute(
            plan(
                "x",
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
        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        self.assertEqual(len(res["discovery_calls"]), 1)
        self.assertIn("quem usa a classe X?", res["discovery_calls"][0]["question"])
        joined = "\n".join(p for p, _ in provider.calls)
        self.assertIn("DISCOVERY FINDING", joined)
        self.assertIn("resultado para: quem usa a classe X?", joined)

    def test_discovery_budget_respected(self):
        provider = FakeProvider(
            [
                discovery_call("q1"),
                discovery_call("q2"),
                write_call("src/x.py", "x\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider, discovery_provider=self._stub())
        result = agent.execute(
            plan(
                "x",
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
            max_discovery_calls=1,
        )
        self.assertEqual(len(result["results"]["TASK-001"]["discovery_calls"]), 1)
        joined = "\n".join(p for p, _ in provider.calls)
        self.assertIn("Discovery budget exhausted", joined)

    def test_scope_growth_replan_required(self):
        provider = FakeProvider(
            [
                discovery_call("onde está o auth?"),
                "REPLAN_REQUIRED: preciso alterar src/extra.py fora do escopo",
            ]
        )
        replan_records = []
        agent = self.make_agent(
            provider,
            discovery_provider=self._stub(extra_files=["src/extra.py"]),
            replanner=replan_records.append,
        )
        result = agent.execute(
            plan(
                "x",
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
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["replans"], ["TASK-001"])
        res = result["results"]["TASK-001"]
        self.assertEqual(res["status"], "REPLAN_REQUIRED")
        self.assertIn("REPLAN_REQUIRED", res["reason"])
        replan = res["replan"]
        self.assertEqual(replan["task_id"], "TASK-001")
        self.assertIn("src/extra.py", replan["files"])
        self.assertIn("scope growth: src/extra.py", replan["evidence"])
        self.assertTrue(replan["for_planner"])
        self.assertEqual(replan_records, [replan])

    def test_replan_via_tool_call(self):
        provider = FakeProvider(
            [
                discovery_call("escopo?"),
                json.dumps(
                    {
                        "tool": "executor",
                        "action": "replan_required",
                        "params": {"reason": "novas dependências arquiteturais"},
                    }
                ),
            ]
        )
        agent = self.make_agent(
            provider, discovery_provider=self._stub(extra_files=["src/extra.py"])
        )
        result = agent.execute(
            plan(
                "x",
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
        self.assertEqual(result["replans"], ["TASK-001"])
        self.assertIn(
            "novas dependências", result["results"]["TASK-001"]["replan"]["reason"]
        )

    def test_discovery_without_provider_or_tool_errors_gracefully(self):
        provider = FakeProvider(
            [
                discovery_call("algo?"),
                write_call("src/x.py", "x\n"),
                "done",
            ]
        )
        registry = build_default_registry()
        agent = ExecutorAgent(
            provider,
            registry,
            root=str(self.root),
            interactor=AutoApproveInteractor(),
        )
        result = agent.execute(
            plan(
                "x",
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
            )
        )
        self.assertEqual(result["status"], "success")
        joined = "\n".join(p for p, _ in provider.calls)
        self.assertIn("discovery tool is not available", joined)


class TestTransitionsAndBuckets(unittest.TestCase):
    def test_stage2_transitions(self):
        legal = [
            (TaskState.EXECUTING, TaskState.NEED_USER_INPUT),
            (TaskState.EXECUTING, TaskState.NEED_CONFIRMATION),
            (TaskState.EXECUTING, TaskState.NEED_DISCOVERY),
            (TaskState.EXECUTING, TaskState.REPLAN_REQUIRED),
            (TaskState.BLOCKED, TaskState.REPLAN_REQUIRED),
            (TaskState.NEED_CONFIRMATION, TaskState.EXECUTING),
            (TaskState.NEED_USER_INPUT, TaskState.EXECUTING),
            (TaskState.NEED_DISCOVERY, TaskState.EXECUTING),
            (TaskState.NEED_CONFIRMATION, TaskState.CANCELLED),
            (TaskState.REPLAN_REQUIRED, TaskState.CANCELLED),
        ]
        for source, target in legal:
            self.assertEqual(
                TaskState.transition(source, target), target, f"{source}->{target}"
            )
        with self.assertRaises(InvalidTransition):
            TaskState.transition(TaskState.NEED_USER_INPUT, TaskState.VALIDATING)

    def test_paused_set_and_blocks_dependents(self):
        self.assertTrue(TaskState.NEED_DISCOVERY in TaskState.PAUSED)
        self.assertTrue(TaskState.NEED_USER_INPUT in TaskState.PAUSED)
        self.assertTrue(TaskState.NEED_CONFIRMATION in TaskState.PAUSED)
        self.assertTrue(TaskState.blocks_dependents(TaskState.REPLAN_REQUIRED))
        self.assertTrue(TaskState.is_valid(TaskState.REPLAN_REQUIRED))

    def test_run_result_buckets(self):
        tasks = [
            {"id": "TASK-001", "title": "a"},
            {"id": "TASK-002", "title": "b"},
        ]
        result = build_run_result(
            plan={"goal": "g", "summary": "s", "tasks": tasks},
            task_order=["TASK-001", "TASK-002"],
            final_states={
                "TASK-001": "REPLAN_REQUIRED",
                "TASK-002": "CANCELLED",
            },
            reasons={"TASK-001": "REPLAN_REQUIRED: escopo", "TASK-002": "cancelado"},
            results={
                "TASK-001": TaskResult(task_id="TASK-001", status="REPLAN_REQUIRED"),
                "TASK-002": TaskResult(task_id="TASK-002", status="CANCELLED"),
            },
            metrics={"tasks_total": 2, "tasks_executed": 0, "tool_calls": 0},
            errors=[],
            warnings=[],
            reason="",
        )
        self.assertEqual(result["replans"], ["TASK-001"])
        self.assertEqual(result["cancelled"], ["TASK-002"])
        self.assertEqual(result["status"], "partial")


class TestAbort(Stage2ExecutorTestCase):
    def test_abort_cancels_paused_and_pending(self):
        provider = FakeProvider([write_call("src/db.py", "db\n"), "done"])
        agent = self.make_agent(provider)
        high = task(
            "migrar",
            "aplicar a migração do banco",
            ["src/db.py"],
            "ctx",
            ["db.py"],
            ["db.py existe"],
            "ev",
        )
        low = task(
            "b", "criar b.py", ["src/b.py"], "ctx", ["b.py"], ["b.py existe"], "ev"
        )
        agent.execute(plan("migrar", [high, low]))
        agent.execute(confirm=True)
        self.assertEqual(agent.pending()["kind"], "confirmation")

        result = agent.abort()
        self.assertEqual(result["status"], "aborted")
        run = result["result"]
        self.assertEqual(sorted(run["cancelled"]), ["TASK-001", "TASK-002"])
        self.assertFalse((self.root / "src/db.py").exists())

    def test_abort_with_nothing_pending(self):
        agent = self.make_agent(FakeProvider([]))
        result = agent.abort()
        self.assertEqual(result["status"], "aborted")


class TestPresenter(Stage2ExecutorTestCase):
    def test_format_plan_preview(self):
        preview = {
            "summary": [
                {
                    "task_id": "TASK-001",
                    "title": "Atualizar tokens",
                    "risk": "LOW",
                    "files": ["src/tokens.css"],
                    "dependencies": [],
                }
            ],
            "impact": "Arquivos potencialmente afetados:\n- src/tokens.css",
            "details": {"goal": "g"},
        }
        rendered = format_plan_preview(preview)
        self.assertIn("TASK-001", rendered)
        self.assertIn("Deseja executar", rendered)

    def test_format_confirmation_and_question(self):
        confirmation = format_confirmation(
            "TASK-001", RiskLevel.CRITICAL, "remove arquivo", ["src/old.py"]
        )
        self.assertIn("TASK-001", confirmation)
        self.assertIn("src/old.py", confirmation)
        question = format_question("TASK-001", "qual banco?", ["sqlite", "postgres"])
        self.assertIn("qual banco?", question)
        self.assertIn("sqlite", question)

    def test_format_result_shows_status_and_discovery(self):
        result = build_run_result(
            plan={
                "goal": "g",
                "summary": "s",
                "tasks": [{"id": "TASK-001", "title": "x"}],
            },
            task_order=["TASK-001"],
            final_states={"TASK-001": "COMPLETED"},
            reasons={"TASK-001": ""},
            results={
                "TASK-001": TaskResult(
                    task_id="TASK-001",
                    status="COMPLETED",
                    changed_files=["src/x.py"],
                    discovery_calls=[{"question": "q1"}],
                )
            },
            metrics={"tasks_total": 1, "tasks_executed": 1, "tool_calls": 1},
            errors=[],
            warnings=[],
            reason="",
        )
        rendered = format_result(result)
        self.assertIn("SUCCESS", rendered)
        self.assertIn("src/x.py", rendered)
        self.assertIn("discovery", rendered)

    def test_summarize_risks(self):
        summary = summarize_risks(
            [
                task("a", "css puro", ["src/a.css"], "c", ["e"], ["f"], "g"),
                task(
                    "b",
                    "deploy do banco",
                    ["src/b.py"],
                    "c",
                    ["e"],
                    ["f"],
                    "g",
                ),
            ]
        )
        self.assertIn("risco máximo", summary)
        self.assertIn("HIGH", summary)


class TestToolSpecStage2(Stage2ExecutorTestCase):
    def test_answer_and_confirm_task_handlers(self):
        provider = FakeProvider(
            [ask_call("escolha?"), write_call("src/x.py", "x\n"), "done"]
        )
        registry = build_default_registry()
        spec = create_executor_tool(provider, registry, root=str(self.root))
        registry.register(spec.name, spec)
        self.assertIn("executor", registry.list_tools())
        out = registry.dispatch(
            "executor",
            "run",
            plan=plan(
                "x",
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
        )
        self.assertEqual(out["status"], "awaiting_confirmation")
        confirmed = registry.dispatch("executor", "execute", confirm=True)
        self.assertEqual(confirmed["status"], "awaiting_confirmation")
        self.assertEqual(confirmed["kind"], "question")
        paused = registry.dispatch("executor", "pending")
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["kind"], "question")
        resumed = registry.dispatch(
            "executor", "answer", task_id="TASK-001", answer="sim"
        )
        self.assertEqual(resumed["status"], "success")


class TestResumePreservesSession(Stage2ExecutorTestCase):
    def test_answer_resume_keeps_tool_history(self):
        provider = FakeProvider(
            [
                ask_call("posso?"),
                write_call("src/a.py", "a\n"),
                write_call("src/b.py", "b\n"),
                "done",
            ]
        )
        agent = self.make_agent(provider)
        agent.execute(
            plan(
                "a+b",
                [
                    task(
                        "t",
                        "criar a e b",
                        ["src/a.py", "src/b.py"],
                        "ctx",
                        ["a.py", "b.py"],
                        ["a e b existem"],
                        "ev",
                    )
                ],
            )
        )
        agent.execute(confirm=True)
        result = agent.answer("TASK-001", "pode")
        self.assertEqual(result["status"], "success")
        res = result["results"]["TASK-001"]
        # The session survived the pause: both writes recorded, one task.
        self.assertEqual(len(res["tool_calls"]), 2)
        self.assertIn("src/a.py", res["changed_files"])
        self.assertIn("src/b.py", res["changed_files"])


if __name__ == "__main__":
    unittest.main()
