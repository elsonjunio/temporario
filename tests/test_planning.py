"""Tests for the Planning Agent subagent: request -> plan with focused
Discovery on demand, atomic tasks with per-task context, dependencies,
readiness/status handling, telemetry and replanning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.discovery import create_discovery_tool
from src.planning import (
    PLANNER_MANUAL,
    PlanningAgent,
    PlanningBudget,
    create_planning_tool,
    normalize_task,
    validate_plan,
)
from src.planning.agent import _is_generic_discovery_question
from src.planning.plan import files_real, resolve_dependencies, vague_criteria
from src.tools.registry import ToolRegistry


class FakeProvider:
    def __init__(self, responses=None, default=None):
        self.responses = list(responses or [])
        self.default = default or (
            "I do not know how to continue. Please give me more instructions."
        )
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        if self.responses:
            return self.responses.pop(0)
        return self.default


DISCOVERY_REPORT = (
    "DISCOVERY RESULT\n\n"
    "TASK\nwhere is the auth flow\n\n"
    "SUMMARY\nAuthentication is in src/auth/service.py.\n\n"
    "RELEVANT FILES\n"
    "1. src/auth/service.py\n"
    "   Purpose: login logic\n"
    "   Relevant symbols: AuthenticationService.authenticate()\n"
    "   Relevance: entry point\n\n"
    "RELATIONSHIPS\nsrc/api/auth.py -> src/auth/service.py\n\n"
    "EVIDENCE\nfind_references found call sites.\n\n"
    "RISKS\nnone.\n\n"
    "NOT INVESTIGATED\ntests\n\n"
    "CONFIDENCE\nhigh"
)


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


def plan_json(goal, tasks):
    return json.dumps({"goal": goal, "summary": "resumo", "tasks": tasks})


class TestPlanValidation(unittest.TestCase):
    def test_normalize_task_accepts_loose_spellings(self):
        t = normalize_task(
            {
                "name": "tokens",
                "description": "atualizar",
                "files": "src/styles.css",
                "context": {"a": "b"},
                "changes": ["x"],
                "acceptance": ["y"],
            },
            index=3,
        )
        self.assertEqual(t["id"], "TASK-003")
        self.assertEqual(t["title"], "tokens")
        self.assertEqual(t["objective"], "atualizar")
        self.assertEqual(t["files"], ["src/styles.css"])
        self.assertEqual(t["context"], "a: b")
        self.assertEqual(t["expected_changes"], ["x"])
        self.assertEqual(t["acceptance_criteria"], ["y"])
        self.assertEqual(t["status"], "READY_FOR_EXECUTION")

    def test_normalize_task_rejects_bad_status(self):
        with self.assertRaises(ValueError):
            normalize_task({"title": "x", "status": "DONE_NOW"})

    def test_files_real(self):
        self.assertTrue(files_real(["src/styles.css"]))
        self.assertTrue(files_real(["index.html"]))
        self.assertFalse(files_real([]))
        self.assertFalse(files_real(["unknown"]))
        self.assertFalse(files_real(["the frontend"]))

    def test_resolve_dependencies_by_title_and_id(self):
        tasks = [
            {"id": "TASK-001", "title": "tokens", "dependencies": []},
            {"id": "TASK-002", "title": "header", "dependencies": ["TASK-001"]},
            {"id": "TASK-003", "title": "card", "dependencies": ["tokens"]},
        ]
        unknown = resolve_dependencies(tasks)
        self.assertEqual(unknown, {})
        self.assertEqual(tasks[2]["dependencies"], ["TASK-001"])

    def test_unknown_dependency_is_issue(self):
        plan = {
            "goal": "x",
            "tasks": [
                {
                    "id": "TASK-002",
                    "title": "b",
                    "objective": "o",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": ["TASK-001"],
                    "files": ["src/b.py"],
                    "context": "c",
                    "expected_changes": ["e"],
                    "acceptance_criteria": ["a"],
                    "evidence": "d",
                }
            ],
        }
        issues, _ = validate_plan(plan)
        self.assertTrue(any("unknown dependencies" in i for i in issues))

    def test_dependency_cycle_is_issue(self):
        plan = {
            "goal": "x",
            "tasks": [
                {
                    "id": "TASK-001",
                    "title": "a",
                    "objective": "o",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": ["TASK-002"],
                    "files": ["src/a.py"],
                    "context": "c",
                    "expected_changes": ["e"],
                    "acceptance_criteria": ["a"],
                    "evidence": "d",
                },
                {
                    "id": "TASK-002",
                    "title": "b",
                    "objective": "o",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": ["TASK-001"],
                    "files": ["src/b.py"],
                    "context": "c",
                    "expected_changes": ["e"],
                    "acceptance_criteria": ["a"],
                    "evidence": "d",
                },
            ],
        }
        issues, _ = validate_plan(plan)
        self.assertTrue(any("cycle" in i for i in issues))

    def test_duplicate_ids_are_issues(self):
        plan = {
            "goal": "x",
            "tasks": [
                {
                    "id": "TASK-001",
                    "title": "a",
                    "objective": "o",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": [],
                    "files": ["src/a.py"],
                    "context": "c",
                    "expected_changes": ["e"],
                    "acceptance_criteria": ["a"],
                    "evidence": "d",
                },
                {
                    "id": "TASK-001",
                    "title": "b",
                    "objective": "o",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": [],
                    "files": ["src/b.py"],
                    "context": "c",
                    "expected_changes": ["e"],
                    "acceptance_criteria": ["a"],
                    "evidence": "d",
                },
            ],
        }
        issues, _ = validate_plan(plan)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_empty_tasks_are_issues(self):
        issues, _ = validate_plan({"goal": "x", "tasks": []})
        self.assertEqual(issues, ["plan must contain a non-empty 'tasks' list"])


class TestPlanningIsolation(unittest.TestCase):
    def test_private_registry_is_read_only(self):
        agent = PlanningAgent(FakeProvider([]), root=".")
        tools = set(agent.readonly_registry.list_tools())
        self.assertFalse(
            tools
            & {"write_file", "patch_file", "delete_file", "move_file", "run_command"}
        )
        self.assertTrue(
            {"read_file", "list_dir", "search_files", "grep_files", "code"} <= tools
        )

    def test_loop_rejects_mutating_tool(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {"file_path": "x"},
                    }
                ),
                plan_json(
                    "x",
                    [
                        task(
                            "a",
                            "o",
                            ["src/a.py"],
                            "c",
                            ["e"],
                            ["a"],
                            "d",
                        )
                    ],
                ),
            ]
        )
        agent = PlanningAgent(provider, root=".")
        result = agent.run("x", max_iterations=4)
        self.assertEqual(result["status"], "success")
        self.assertIn("write_file", provider.calls[1][0])
        self.assertTrue(
            any(
                "read-only" in prompt or "unknown tool" in prompt
                for _, prompt in provider.calls[:2]
            )
        )


class TestPlanningAgent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _agent(self, provider, **kw):
        return PlanningAgent(provider, root=str(self.root), **kw)

    # 1. requisição simples (sem discovery)
    def test_simple_request_final_plan(self):
        plan = plan_json(
            "aplicar DESIGN.md",
            [
                task(
                    "tokens globais",
                    "atualizar tokens globais",
                    ["src/styles.css"],
                    "--color-background está na linha 12; DESIGN.md define o novo valor.",
                    ["trocar o valor de --color-background"],
                    ["--color-background usa o valor de DESIGN.md"],
                    "DESIGN.md define os valores.",
                )
            ],
        )
        provider = FakeProvider([plan])
        result = self._agent(provider).run("aplicar DESIGN.md")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["discovery_calls"], 0)
        tasks = result["plan"]["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["status"], "READY_FOR_EXECUTION")

    # 2. requisição que exige Discovery
    def test_request_requiring_discovery(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "discovery",
                        "action": "run",
                        "params": {
                            "question": "Onde é AuthenticationService?",
                            "scope": "src",
                        },
                    }
                ),
                DISCOVERY_REPORT,
                plan_json(
                    "refactor auth",
                    [
                        task(
                            "mover autenticação",
                            "mover AuthenticationService",
                            ["src/auth/service.py"],
                            "AuthenticationService está em service.py; api/auth.py o usa.",
                            ["mover a classe"],
                            ["imports em src/api/auth.py continuam válidos"],
                            "discovery: service.py implementa, api/auth.py importa.",
                        )
                    ],
                ),
            ]
        )
        result = self._agent(provider).run("refactor auth")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["discovery_calls"], 1)
        self.assertEqual(result["metrics"]["discovery_revisits"], 0)
        self.assertEqual(result["metrics"]["planning_iterations"], 2)
        self.assertIn("service.py", result["plan"]["tasks"][0]["files"][0])

    # 3. múltiplas chamadas ao Discovery
    def test_multiple_discovery_calls(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "discovery",
                        "action": "run",
                        "params": {"question": "onde está o auth?"},
                    }
                ),
                DISCOVERY_REPORT,
                json.dumps(
                    {
                        "tool": "discovery",
                        "action": "run",
                        "params": {"question": "quem usa AuthenticationService?"},
                    }
                ),
                DISCOVERY_REPORT,
                plan_json(
                    "refactor auth",
                    [
                        task(
                            "mover autenticação",
                            "mover AuthenticationService",
                            ["src/auth/service.py"],
                            "service.py implementa.",
                            ["mover"],
                            ["imports válidos"],
                            "discovery: implementação e usuários localizados.",
                        )
                    ],
                ),
            ]
        )
        result = self._agent(provider).run("refactor auth")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["discovery_calls"], 2)
        self.assertEqual(result["metrics"]["discovery_revisits"], 1)

    # 4. criação de múltiplas tarefas
    def test_multiple_tasks(self):
        plan = plan_json(
            "aplicar DESIGN.md",
            [
                task(
                    "tokens",
                    "atualizar tokens",
                    ["src/styles.css"],
                    "c1",
                    ["e1"],
                    ["a1"],
                    "d1",
                ),
                task(
                    "fontes",
                    "atualizar fontes",
                    ["index.html"],
                    "c2",
                    ["e2"],
                    ["a2"],
                    "d2",
                ),
                task(
                    "header",
                    "migrar Header",
                    ["src/Header.jsx"],
                    "c3",
                    ["e3"],
                    ["a3"],
                    "d3",
                ),
            ],
        )
        result = self._agent(FakeProvider([plan])).run("design")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["tasks_created"], 3)
        self.assertEqual(result["metrics"]["tasks_ready"], 3)

    # 5. tarefas dependentes
    def test_dependent_tasks(self):
        plan = plan_json(
            "migrar componentes",
            [
                task(
                    "tokens",
                    "criar tokens",
                    ["src/styles.css"],
                    "c1",
                    ["e1"],
                    ["a1"],
                    "d1",
                ),
                task(
                    "header",
                    "migrar Header para tokens",
                    ["src/Header.jsx"],
                    "c2",
                    ["e2"],
                    ["a2"],
                    "d2",
                    dependencies=["TASK-001"],
                ),
                task(
                    "card",
                    "migrar ProductCard para tokens",
                    ["src/ProductCard.jsx"],
                    "c3",
                    ["e3"],
                    ["a3"],
                    "d3",
                    dependencies=["TASK-001"],
                ),
            ],
        )
        result = self._agent(FakeProvider([plan])).run("migrar")
        tasks = result["plan"]["tasks"]
        self.assertEqual(tasks[1]["dependencies"], ["TASK-001"])
        self.assertEqual(tasks[2]["dependencies"], ["TASK-001"])
        self.assertEqual(result["status"], "success")

    # 6. tarefas independentes
    def test_independent_tasks(self):
        plan = plan_json(
            "duas mudanças",
            [
                task("a", "mudança A", ["src/a.py"], "c1", ["e1"], ["a1"], "d1"),
                task("b", "mudança B", ["src/b.py"], "c2", ["e2"], ["a2"], "d2"),
            ],
        )
        result = self._agent(FakeProvider([plan])).run("duas")
        tasks = result["plan"]["tasks"]
        self.assertEqual(tasks[0]["dependencies"], [])
        self.assertEqual(tasks[1]["dependencies"], [])
        self.assertEqual(result["status"], "success")

    # 7. tarefa insuficientemente especificada -> downgrade
    def test_insufficiently_specified_task_is_downgraded(self):
        plan = plan_json(
            "estilos",
            [
                {
                    "title": "sistema de estilos",
                    "objective": "corrigir estilos",
                    "status": "READY_FOR_EXECUTION",
                    "dependencies": [],
                    "files": [],
                    "context": "",
                    "expected_changes": [],
                    "acceptance_criteria": ["fica correto"],
                    "evidence": "DESIGN.md",
                }
            ],
        )
        result = self._agent(FakeProvider([plan])).run("estilos")
        self.assertEqual(result["status"], "partial")
        tasks = result["plan"]["tasks"]
        self.assertEqual(tasks[0]["status"], "DISCOVERY_REQUIRED")
        self.assertTrue(
            any(
                "downgraded" in w or "not ready" in w
                for w in result["plan"]["warnings"]
            )
        )
        self.assertEqual(result["metrics"]["tasks_discovery_required"], 1)
        self.assertEqual(result["metrics"]["tasks_ready"], 0)

    # 8. tarefa que deve permanecer DISCOVERY_REQUIRED
    def test_task_stays_discovery_required(self):
        plan = plan_json(
            "migração",
            [
                task(
                    "parte desconhecida",
                    "migrar parte ainda não investigada",
                    [],
                    "",
                    [],
                    [],
                    "",
                    status="DISCOVERY_REQUIRED",
                )
            ],
        )
        result = self._agent(FakeProvider([plan])).run("migração")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["plan"]["tasks"][0]["status"], "DISCOVERY_REQUIRED")

    # 9. contexto específico por tarefa (isolamento)
    def test_task_specific_context_is_isolated(self):
        plan = plan_json(
            "aplicar DESIGN.md",
            [
                task(
                    "tokens globais",
                    "atualizar tokens de cor",
                    ["src/styles.css"],
                    "--color-background na linha 12; DESIGN.md define novos valores.",
                    ["trocar --color-background"],
                    ["--color-background usa DESIGN.md"],
                    "DESIGN.md define os valores; styles.css os declara.",
                ),
                task(
                    "carregamento de fontes",
                    "adicionar Playfair Display e DM Sans",
                    ["index.html"],
                    "index.html carrega Arial hoje; DESIGN.md pede Playfair Display e DM Sans.",
                    ["incluir as duas fontes"],
                    ["Playfair Display e DM Sans carregam em index.html"],
                    "DESIGN.md especifica as fontes.",
                ),
            ],
        )
        result = self._agent(FakeProvider([plan])).run("design")
        first, second = result["plan"]["tasks"]
        self.assertIn("--color-background", first["context"])
        self.assertNotIn("--color-background", second["context"])
        self.assertNotIn("Playfair", first["context"])
        self.assertIn("Playfair", second["context"])
        self.assertEqual(first["files"], ["src/styles.css"])
        self.assertEqual(second["files"], ["index.html"])

    # 10. critérios de aceitação objetivos e verificáveis
    def test_acceptance_criteria_are_verifiable(self):
        plan = plan_json(
            "autenticação",
            [
                task(
                    "auth",
                    "refatorar autenticação",
                    ["src/auth/service.py"],
                    "c",
                    ["extrair validate()"],
                    [
                        "validate() existe em AuthenticationService",
                        "os testes existentes de autenticação continuam passando",
                    ],
                    "d",
                )
            ],
        )
        result = self._agent(FakeProvider([plan])).run("auth")
        criteria = result["plan"]["tasks"][0]["acceptance_criteria"]
        self.assertEqual(len(criteria), 2)
        self.assertIn("validate()", criteria[0])
        self.assertIn("testes", criteria[1])
        self.assertNotIn("código deve ficar correto", criteria)

    # 11. replanning após resultado de execução
    def test_replan_after_execution(self):
        previous = {
            "goal": "aplicar DESIGN.md",
            "summary": "s",
            "tasks": [
                task(
                    "tokens globais",
                    "criar tokens",
                    ["src/styles.css"],
                    "c1",
                    ["e1"],
                    ["a1"],
                    "d1",
                )
            ],
        }
        increment = plan_json(
            "aplicar DESIGN.md",
            [
                task(
                    "migrar Header",
                    "header usa os tokens",
                    ["src/Header.jsx"],
                    "Header usa cores hardcoded hoje.",
                    ["trocar por tokens"],
                    ["Header usa --color-primary"],
                    "tokens criados em TASK-001.",
                    dependencies=["TASK-001"],
                )
            ],
        )
        provider = FakeProvider([increment])
        result = self._agent(provider).replan(
            "aplicar DESIGN.md",
            previous_plan=previous,
            task_results=[{"id": "TASK-001", "status": "success"}],
        )
        self.assertTrue(result["replan"])
        by_id = {t["id"]: t for t in result["plan"]["tasks"]}
        self.assertEqual(by_id["TASK-001"]["status"], "COMPLETED")
        self.assertEqual(by_id["TASK-002"]["status"], "READY_FOR_EXECUTION")
        self.assertEqual(by_id["TASK-002"]["dependencies"], ["TASK-001"])
        self.assertIn("REPLAN CONTEXT", provider.calls[0][1])

    def test_replan_marks_failed_task(self):
        previous = {
            "goal": "x",
            "tasks": [
                task(
                    "header",
                    "migrar Header",
                    ["src/Header.jsx"],
                    "c",
                    ["e"],
                    ["a"],
                    "d",
                )
            ],
        }
        revised = plan_json(
            "x",
            [
                task(
                    "header",
                    "migrar Header",
                    ["src/Header.jsx"],
                    "contexto revisado",
                    ["e"],
                    ["a"],
                    "d",
                    notes="replanned",
                )
            ],
        )
        provider = FakeProvider([revised])
        result = self._agent(provider).replan(
            "x",
            previous_plan=previous,
            task_results={"TASK-001": {"status": "failed", "notes": "anchor moved"}},
        )
        self.assertIn("anchor moved", result["plan"]["tasks"][0]["notes"])

    def test_requires_request(self):
        result = self._agent(FakeProvider([])).run("")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_discovery_budget_exhausted(self):
        q = lambda i: json.dumps(
            {"tool": "discovery", "action": "run", "params": {"question": f"q{i}"}}
        )
        provider = FakeProvider([q(1), DISCOVERY_REPORT, q(2), DISCOVERY_REPORT, q(3)])
        result = self._agent(provider).run("x", max_discovery_calls=2, max_iterations=6)
        self.assertEqual(result["metrics"]["discovery_calls"], 2)
        self.assertIn("discovery budget", provider.calls[5][0])
        self.assertEqual(result["status"], "partial")

    def test_duplicate_discovery_question_rejected(self):
        q = json.dumps(
            {
                "tool": "discovery",
                "action": "run",
                "params": {"question": "mesma pergunta"},
            }
        )
        provider = FakeProvider([q, DISCOVERY_REPORT, q])
        result = self._agent(provider).run("x", max_iterations=4)
        self.assertEqual(result["metrics"]["discovery_calls"], 1)
        self.assertIn("already asked", provider.calls[3][0])

    def test_max_tasks_guard(self):
        task_call = json.dumps(
            {
                "tool": "planner",
                "action": "add_task",
                "params": task("t", "o", ["src/a.py"], "c", ["e"], ["a"], "d"),
            }
        )
        provider = FakeProvider([task_call, task_call, task_call])
        result = self._agent(provider).run("x", max_tasks=1, max_iterations=4)
        self.assertEqual(result["metrics"]["tasks_created"], 1)
        self.assertIn("max tasks reached", provider.calls[2][0])

    def test_incremental_add_task_then_budget_assembles_plan(self):
        t1 = json.dumps(
            {
                "tool": "planner",
                "action": "add_task",
                "params": task("a", "o", ["src/a.py"], "c1", ["e1"], ["a1"], "d1"),
            }
        )
        t2 = json.dumps(
            {
                "tool": "planner",
                "action": "add_task",
                "params": task("b", "o", ["src/b.py"], "c2", ["e2"], ["a2"], "d2"),
            }
        )
        provider = FakeProvider([t1, t2])
        result = self._agent(provider).run("x", max_iterations=2)
        self.assertEqual(result["metrics"]["tasks_created"], 2)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["plan"]["tasks"]), 2)

    def test_incremental_discovery_then_task(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "discovery",
                        "action": "run",
                        "params": {"question": "onde?"},
                    }
                ),
                DISCOVERY_REPORT,
                json.dumps(
                    {
                        "tool": "planner",
                        "action": "add_task",
                        "params": task(
                            "auth",
                            "mover",
                            ["src/auth/service.py"],
                            "c",
                            ["e"],
                            ["a"],
                            "evidência do discovery",
                        ),
                    }
                ),
            ]
        )
        result = self._agent(provider).run("x", max_iterations=3)
        self.assertEqual(result["metrics"]["discovery_calls"], 1)
        self.assertEqual(result["metrics"]["tasks_created"], 1)
        self.assertEqual(result["plan"]["tasks"][0]["status"], "READY_FOR_EXECUTION")

    def test_read_only_analysis_in_loop(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text("x = 1\n")
        plan = plan_json(
            "x",
            [
                task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d"),
            ],
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": "src/a.py"},
                    }
                ),
                plan,
            ]
        )
        result = self._agent(provider).run("x")
        self.assertEqual(result["status"], "success")
        self.assertIn("x = 1", str(provider.calls[1][0]) or "read_file")
        self.assertEqual(result["metrics"]["planning_iterations"], 2)

    def test_telemetry_keys_present(self):
        plan = plan_json(
            "x",
            [task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d")],
        )
        result = self._agent(FakeProvider([plan])).run("x")
        metrics = result["metrics"]
        for key in (
            "planning_iterations",
            "discovery_calls",
            "discovery_revisits",
            "tasks_created",
            "tasks_ready",
            "tasks_blocked",
            "tasks_discovery_required",
            "tasks_completed",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "tokens_used",
            "final_plan_tokens",
            "planning_time",
        ):
            self.assertIn(key, metrics)
        self.assertGreater(metrics["final_plan_tokens"], 0)
        self.assertGreaterEqual(metrics["planning_time"], 0)

    def test_discovery_revisits_telemetry(self):
        plan = plan_json(
            "x",
            [task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d")],
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {"tool": "discovery", "action": "run", "params": {"question": "q1"}}
                ),
                DISCOVERY_REPORT,
                json.dumps(
                    {"tool": "discovery", "action": "run", "params": {"question": "q2"}}
                ),
                DISCOVERY_REPORT,
                plan,
            ]
        )
        result = self._agent(provider).run("x")
        self.assertEqual(result["metrics"]["discovery_calls"], 2)
        self.assertEqual(result["metrics"]["discovery_revisits"], 1)


class TestPlanningTool(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_manual_and_handlers(self):
        tool = create_planning_tool(FakeProvider([]), root=str(self.root))
        self.assertEqual(tool.name, "planner")
        self.assertIn("run", tool.get_manual())
        self.assertIn("discover", tool.get_manual())
        self.assertIn("replan", tool.get_manual())
        result = tool.dispatch("run", request="")
        self.assertEqual(result["status"], "error")

    def test_unknown_action(self):
        tool = create_planning_tool(FakeProvider([]), root=str(self.root))
        result = tool.dispatch("nope")
        self.assertEqual(result["error"], "unknown_action")

    def test_public_discover_delegates_to_discovery(self):
        provider = FakeProvider([DISCOVERY_REPORT])
        tool = create_planning_tool(provider, root=str(self.root))
        result = tool.dispatch("discover", question="onde está o auth?")
        self.assertEqual(result["status"], "discovery_done")
        self.assertEqual(result["metrics"]["discovery_calls"], 1)
        self.assertIn("service.py", result["result"]["result"]["report"])

    def test_public_discover_rejects_generic_question(self):
        tool = create_planning_tool(FakeProvider([]), root=str(self.root))
        result = tool.dispatch("discover", question="analise o projeto inteiro")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")
        self.assertTrue(result["error"]["recoverable"])
        self.assertIn("too generic", result["message"])

    def test_replan_carries_executor_discovery_into_context(self):
        previous = {
            "goal": "x",
            "tasks": [
                task(
                    "t1",
                    "o",
                    ["src/a.py"],
                    "c",
                    ["e"],
                    ["a"],
                    "d",
                    id="TASK-001",
                )
            ],
        }
        revised = plan_json(
            "x",
            [
                task(
                    "t1",
                    "o2",
                    ["src/a.py"],
                    "c2",
                    ["e2"],
                    ["a2"],
                    "d2",
                )
            ],
        )
        provider = FakeProvider([revised])
        result = create_planning_tool(provider, root=str(self.root)).dispatch(
            "replan",
            request="x",
            previous_plan=previous,
            task_results={
                "TASK-001": {
                    "status": "failed",
                    "discovery_calls": [
                        {"question": "onde fica o AuthService?", "status": "success"}
                    ],
                }
            },
        )
        self.assertEqual(result["status"], "success")
        self.assertIn("REPLAN CONTEXT", provider.calls[0][1])
        self.assertIn("AuthService", provider.calls[0][1])

    def test_pending_returns_last_plan(self):
        provider = FakeProvider(
            [
                plan_json(
                    "x",
                    [task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d")],
                )
            ]
        )
        tool = create_planning_tool(provider, root=str(self.root))
        run = tool.dispatch("run", request="x")
        self.assertEqual(run["status"], "success")
        pending = tool.dispatch("pending")
        self.assertEqual(pending["status"], "pending_plan")
        self.assertEqual(len(pending["plan"]["tasks"]), 1)

    def test_reuses_registry_discovery_tool(self):
        registry = ToolRegistry()
        registry.register(
            "discovery",
            create_discovery_tool(
                FakeProvider([DISCOVERY_REPORT]), root=str(self.root)
            ),
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {"tool": "discovery", "action": "run", "params": {"question": "q"}}
                ),
                plan_json(
                    "x",
                    [task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d")],
                ),
            ]
        )
        tool = create_planning_tool(provider, registry, root=str(self.root))
        result = tool.dispatch("run", request="x")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["discovery_calls"], 1)


class TestPlanningBudget(unittest.TestCase):
    def test_apply_overrides(self):
        budget = PlanningBudget(max_iterations=1, max_discovery_calls=1, max_tasks=1)
        self.assertEqual(budget.apply_overrides(max_tasks=3).max_tasks, 3)
        self.assertEqual(budget.apply_overrides(nope=4).max_tasks, 1)
        self.assertEqual(budget.to_dict()["max_iterations"], 1)

    def test_context_budget_stops_early(self):
        plan = plan_json(
            "x",
            [task("a", "o", ["src/a.py"], "c", ["e"], ["a"], "d")],
        )
        provider = FakeProvider([plan])
        agent = PlanningAgent(provider, root=".")
        result = agent.run("x", max_context_tokens=1)
        self.assertEqual(result["status"], "partial")
        self.assertIn("context exceeded", result["plan"]["reason"])
        self.assertEqual(provider.calls, [])


class TestPlanningAdjustments(unittest.TestCase):
    """Guards added from the 30-characteristic review: generic discovery
    questions, vague acceptance criteria and executor-discovery feedback."""

    def test_generic_discovery_question_detector(self):
        for question in (
            "analise o projeto inteiro",
            "o que precisa ser feito?",
            "explore o codebase",
            "entenda o código",
            "descubra",
            "explore",
            "what should i do",
        ):
            self.assertTrue(_is_generic_discovery_question(question), question)
        for question in (
            "onde é AuthenticationService?",
            "quais arquivos importam src/auth.py?",
            "como o fluxo de login funciona em src/auth/service.py?",
        ):
            self.assertFalse(_is_generic_discovery_question(question), question)

    def test_vague_criteria_detector(self):
        self.assertIn(
            "deixar funcionando",
            vague_criteria(["deixar funcionando", "corrigir o problema"]),
        )
        self.assertEqual(vague_criteria(["executar pytest -q", "sem falhas"]), [])

    def test_validate_plan_warns_on_vague_criteria(self):
        plan = {
            "goal": "x",
            "tasks": [
                task(
                    "a",
                    "o",
                    ["src/a.py"],
                    "c",
                    ["e"],
                    ["deixar funcionando"],
                    "d",
                    id="TASK-001",
                )
            ],
        }
        issues, warnings = validate_plan(plan)
        self.assertEqual(issues, [])
        self.assertEqual(plan["tasks"][0]["status"], "READY_FOR_EXECUTION")
        self.assertTrue(any("vague acceptance criteria" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
