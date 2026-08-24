import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.orchestrator import create_orchestrator_tool
from src.orchestrator.discovery import Discovery
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.patchgen import PatchGenerator
from src.orchestrator.preassessment import Decomposer, PreAssessor, score_request
from src.orchestrator.undo import UndoLog
from src.tools.base import ToolSpec
from src.tools.registry import build_default_registry


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        if self.responses:
            self.last = self.responses.pop(0)
            return self.last
        return self.last


class OrchestratorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.registry = build_default_registry()
        # Force the legacy confirmation flow so tests never block on the
        # interactive input() gate, regardless of whether the suite runs in a
        # TTY. The interactive-gate tests below opt back in explicitly.
        self._confirm_mode_patcher = patch.dict(
            os.environ, {"ORCH_CONFIRM_MODE": "agent"}
        )
        self._confirm_mode_patcher.start()
        self.addCleanup(self._confirm_mode_patcher.stop)

    def _make_orch(self, provider):
        return Orchestrator(
            self.registry,
            provider,
            root=str(self.root),
        )

    def _make_tool(self, provider):
        return create_orchestrator_tool(
            self.registry,
            provider,
            root=str(self.root),
        )

    def _register_discovery(self):
        """Register a discovery subagent so the orchestrator's fallback gate
        (which only fires when the subagent is actually available, i.e.
        balanced mode) opens. Fast mode leaves it unregistered."""
        self.registry.register(
            "discovery",
            ToolSpec(
                name="discovery",
                handlers={"run": lambda **kwargs: {"status": "success"}},
                manual="",
            ),
        )


class TestDiscovery(OrchestratorTestCase):
    def test_finds_target(self):
        (self.root / "notes.txt").write_text("foo bar\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("change something in notes.txt")
        self.assertEqual(result["status"], "ok")
        paths = [c["path"] for c in result["candidates"]]
        self.assertTrue(any("notes.txt" in p for p in paths))

    def test_poor_when_no_match(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("xyzzy zorp unknown")
        self.assertEqual(result["status"], "poor")

    def test_empty_workspace_returns_greenfield_evidence(self):
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover(
            "crie uma aplicação web completa com backend e frontend"
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["greenfield"])
        candidate = result["candidates"][0]
        self.assertEqual(candidate["type"], "workspace_root")
        self.assertTrue(candidate["new_project"])
        self.assertEqual(candidate["path"], str(self.root.resolve()))

    def test_dotfiles_only_workspace_counts_as_empty(self):
        (self.root / ".git").mkdir()
        (self.root / ".gitignore").write_text("x\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("build a whole new project")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["greenfield"])


class TestDiscoveryFallback(OrchestratorTestCase):
    """The orchestrator's discovery fallback (balanced mode) fills gaps when
    the internal keyword discovery returns weak evidence."""

    def test_fallback_used_when_internal_poor(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        new_path = self.root / "generated" / "out.txt"
        calls = []

        def fallback(request, terms, paths):
            calls.append((request, terms, paths))
            return {
                "status": "ok",
                "request": request,
                "terms": terms or [],
                "reason": "filled by subagent",
                "count": 1,
                "candidates": [
                    {
                        "name": "out.txt",
                        "path": str(new_path),
                        "type": "new_file",
                        "snippet": "",
                    }
                ],
                "source": "discovery_subagent",
            }

        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(new_path),
                                    "content": "x",
                                },
                                "validate_after": True,
                                "description": "write out.txt",
                            }
                        ]
                    }
                )
            ]
        )
        orch = Orchestrator(
            self.registry,
            provider,
            root=str(self.root),
            discovery_fallback=fallback,
        )
        self._register_discovery()
        plan = orch.plan("create something unusual")
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "create something unusual")
        evidence = orch.last_evidence
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["source"], "discovery_subagent")

    def test_fallback_not_invoked_when_internal_ok(self):
        (self.root / "notes.txt").write_text("hi\n")
        calls = []

        def fallback(request, terms, paths):
            calls.append(request)
            return {"status": "poor", "request": request, "count": 0, "candidates": []}

        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "read_file",
                                "action": "read",
                                "params": {"file_path": str(self.root / "notes.txt")},
                                "description": "read notes",
                            }
                        ]
                    }
                )
            ]
        )
        orch = Orchestrator(
            self.registry,
            provider,
            root=str(self.root),
            discovery_fallback=fallback,
        )
        plan = orch.plan("change something in notes.txt")
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(calls, [])

    def test_fallback_poor_still_aborts(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        calls = []

        def fallback(request, terms, paths):
            calls.append(request)
            return {
                "status": "poor",
                "request": request,
                "reason": "still nothing found",
                "count": 0,
                "candidates": [],
            }

        provider = FakeProvider()
        orch = Orchestrator(
            self.registry,
            provider,
            root=str(self.root),
            discovery_fallback=fallback,
        )
        self._register_discovery()
        plan = orch.plan("qwerty zorp nope")
        self.assertEqual(plan["status"], "aborted")
        self.assertEqual(plan["reason"], "still nothing found")
        self.assertEqual(len(calls), 1)

    def test_discover_uses_fallback_too(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        calls = []

        def fallback(request, terms, paths):
            calls.append(request)
            return {
                "status": "ok",
                "request": request,
                "count": 0,
                "candidates": [],
            }

        orch = Orchestrator(
            self.registry,
            FakeProvider(),
            root=str(self.root),
            discovery_fallback=fallback,
        )
        self._register_discovery()
        result = orch.discover("qwerty zorp nope")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 1)

    def test_create_orchestrator_tool_passes_fallback(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        calls = []

        def fallback(request, terms, paths):
            calls.append(request)
            return {"status": "ok", "request": request, "count": 0, "candidates": []}

        tool = create_orchestrator_tool(
            self.registry,
            FakeProvider(),
            root=str(self.root),
            discovery_fallback=fallback,
        )
        self._register_discovery()
        result = tool.dispatch("discover", request="qwerty zorp nope")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 1)

    def test_explicit_terms(self):
        (self.root / "target.txt").write_text("hi\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("do something", terms=["target"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 1)


class TestDiscoveryGreenfield(OrchestratorTestCase):
    def test_seed_paths_creates_new_file_candidates(self):
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover(
            "create the backend skeleton",
            seed_paths=["backend/app/main.py", "frontend/package.json"],
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 2)
        types = {c["path"]: c["type"] for c in result["candidates"]}
        self.assertEqual(
            types[str((self.root / "backend/app/main.py").resolve())],
            "new_file",
        )
        self.assertEqual(
            types[str((self.root / "frontend/package.json").resolve())],
            "new_file",
        )
        self.assertIn("seeded", result["reason"])

    def test_seed_paths_marks_existing_paths_as_files(self):
        (self.root / "README.md").write_text("hi\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("edit readme", seed_paths=["README.md"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["candidates"][0]["type"], "file")

    def test_seed_paths_dedupes(self):
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("x", seed_paths=["a.txt", "a.txt", "b.txt"])
        self.assertEqual(result["count"], 2)


class TestOrchestratorGreenfield(OrchestratorTestCase):
    def _steps_json(self, *file_paths):
        steps = []
        for path in file_paths:
            steps.append(
                {
                    "tool": "write_file",
                    "action": "write",
                    "params": {"file_path": str(path), "content": "x"},
                    "validate_after": True,
                    "description": f"create {path.name}",
                }
            )
        return json.dumps({"steps": steps})

    def test_plan_with_paths_bypasses_keyword_search(self):
        provider = FakeProvider([self._steps_json(self.root / "backend" / "main.py")])
        orch = self._make_orch(provider)
        plan = orch.plan(
            "create backend skeleton",
            paths=["backend/main.py", "backend/requirements.txt"],
        )
        self.assertEqual(plan["status"], "ok")
        evidence = orch.last_evidence
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["count"], 2)
        self.assertTrue(all(c["type"] == "new_file" for c in evidence["candidates"]))

    def test_run_with_paths_returns_awaiting_confirmation_with_impact(self):
        provider = FakeProvider([self._steps_json(self.root / "backend" / "main.py")])
        orch = self._make_orch(provider)
        result = orch.run("create backend skeleton", paths=["backend/main.py"])
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertIn("Impact analysis", result["impact"])
        self.assertIn("mode=new", result["impact"])


class TestPlanner(OrchestratorTestCase):
    def test_parses_plan(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(self.root / "a.txt"),
                                    "content": "x",
                                },
                                "validate_after": True,
                                "description": "write a.txt",
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_orch(provider)
        evidence = {
            "status": "ok",
            "request": "r",
            "terms": ["a"],
            "count": 1,
            "candidates": [],
        }
        plan = orch.planner.plan("write a.txt", evidence)
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["steps"][0]["tool"], "write_file")

    def test_rejects_unknown_tool(self):
        provider = FakeProvider(
            [json.dumps({"steps": [{"tool": "ghost", "action": "x", "params": {}}]})]
        )
        orch = self._make_orch(provider)
        evidence = {
            "status": "ok",
            "request": "r",
            "terms": [],
            "count": 1,
            "candidates": [],
        }
        plan = orch.planner.plan("r", evidence)
        self.assertEqual(plan["status"], "error")
        self.assertIn("ghost", plan["message"])

    def test_rejects_non_json(self):
        provider = FakeProvider(["no json here"])
        orch = self._make_orch(provider)
        evidence = {
            "status": "ok",
            "request": "r",
            "terms": [],
            "count": 1,
            "candidates": [],
        }
        plan = orch.planner.plan("r", evidence)
        self.assertEqual(plan["status"], "error")


class TestExecutor(OrchestratorTestCase):
    def test_executes_plan(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "out.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "validate_after": True,
                "expect": "hello",
                "description": "create out.txt",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])
        self.assertEqual(target.read_text(), "hello\n")

    def test_execute_requires_confirmation(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "out.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
            }
        ]
        result = orch.execute(steps)
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(len(result["summary"]), 1)
        self.assertFalse(target.exists())

    def test_run_command_defaults_cwd_to_root(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": "pwd"},
                "validate_after": True,
                "expect": str(self.root),
                "description": "print working dir",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])

    def test_run_command_expect_is_case_insensitive(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": 'echo "Reinitialized existing repository"'},
                "validate_after": True,
                "expect": "Initialized",
                "description": "re-init repo",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])

    def test_write_file_expect_result_mode_passes(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "x.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "validate_after": True,
                "expect": "created",
                "description": "create x.txt",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])
        self.assertEqual(target.read_text(), "hello\n")

    def test_write_file_missing_expect_warns_not_fails(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "auth.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {
                    "file_path": str(target),
                    "content": 'router = APIRouter(prefix="/api/auth")\n@router.post("/register")\n',
                },
                "validate_after": True,
                "expect": "/api/auth/register",
                "description": "composed path not verbatim",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])
        self.assertIn(
            "not found verbatim", result["trace"][0]["validated"]["expect_warn"]
        )

    def test_read_missing_file_warns_not_fails(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "read_file",
                "action": "read",
                "params": {"file_path": str(self.root / "settings.py")},
                "validate_after": False,
                "description": "read settings",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])
        self.assertIn("may not exist", result["trace"][0]["read_warn"])

    def test_command_expect_missing_warns_not_fails(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": 'echo "Seed concluido com sucesso"'},
                "validate_after": True,
                "expect": "seed_concluido",
                "description": "run seed",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["trace"][0]["ok"])
        self.assertIn(
            "not found in command output",
            result["trace"][0]["validated"]["expect_warn"],
        )

    def test_validation_expect_failure_rolls_back(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "out.txt"
        target.write_text("original\n")
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "original\n"},
                "validate_after": True,
                "expect": "WRONG",
                "description": "rewrite out.txt",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rollback"]["status"], "rolled_back")
        self.assertEqual(target.read_text(), "original\n")

    def test_failed_step_rolls_back_previous(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "out.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "original\n"},
            },
            {
                "tool": "patch_file",
                "action": "apply",
                "params": {
                    "file_path": str(target),
                    "diff": "@@ -1,1 +1,1 @@\n-nope\n+yes\n",
                },
            },
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_step"]["tool"], "patch_file")
        self.assertEqual(result["rollback"]["status"], "rolled_back")
        self.assertFalse(target.exists())


class TestPlanAnchorFeedback(OrchestratorTestCase):
    def test_plan_retries_when_anchor_missing(self):
        target = str(self.root / "notes.txt")
        (self.root / "notes.txt").write_text("hello\nworld\n")
        bad = json.dumps(
            {
                "steps": [
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": target},
                        "description": "read notes",
                    },
                    {
                        "tool": "patch_file",
                        "action": "replace",
                        "params": {"file_path": target, "old": "invented", "new": "x"},
                        "description": "patch notes",
                    },
                ]
            }
        )
        good = json.dumps(
            {
                "steps": [
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": target},
                        "description": "read notes",
                    },
                    {
                        "tool": "patch_file",
                        "action": "replace",
                        "params": {"file_path": target, "old": "world", "new": "x"},
                        "description": "patch notes",
                    },
                ]
            }
        )
        provider = FakeProvider([bad, good])
        orch = self._make_orch(provider)
        plan = orch.plan("change notes", terms=["notes"])
        self.assertEqual(plan["status"], "ok")
        self.assertNotIn("REJECTION FEEDBACK", provider.calls[0][0])
        self.assertIn("REJECTION FEEDBACK", provider.calls[1][0])
        self.assertEqual(plan["steps"][1]["params"]["old"], "world")

    def test_plan_error_when_anchor_never_fixed(self):
        target = str(self.root / "notes.txt")
        (self.root / "notes.txt").write_text("hello\nworld\n")
        bad = json.dumps(
            {
                "steps": [
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": target},
                        "description": "read notes",
                    },
                    {
                        "tool": "patch_file",
                        "action": "replace",
                        "params": {"file_path": target, "old": "invented", "new": "x"},
                        "description": "patch notes",
                    },
                ]
            }
        )
        provider = FakeProvider([bad, bad])
        orch = self._make_orch(provider)
        result = orch.plan("change notes", terms=["notes"])
        self.assertEqual(result["status"], "error")
        self.assertIn("old", result["message"])


class TestExecRetry(OrchestratorTestCase):
    def test_retry_replans_and_succeeds_after_patch_failure(self):
        target = str(self.root / "notes.txt")
        provider = FakeProvider()
        orch = self._make_orch(provider)
        orch.last_request = "fix notes"
        notes_path = self.root / "notes.txt"
        orch.last_evidence = {
            "status": "ok",
            "request": "fix notes",
            "terms": [],
            "count": 1,
            "candidates": [
                {"name": "notes.txt", "path": target, "type": "file", "snippet": ""}
            ],
        }
        orch.last_impact = {"request": "fix notes", "targets": []}
        fixed = json.dumps(
            {
                "steps": [
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {"file_path": target, "content": "hello\nworld\n"},
                        "description": "create notes",
                    },
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": target},
                        "description": "read notes",
                    },
                    {
                        "tool": "patch_file",
                        "action": "replace",
                        "params": {
                            "file_path": target,
                            "old": "world",
                            "new": "everyone",
                        },
                        "description": "patch notes",
                    },
                ]
            }
        )
        provider.responses = [fixed]
        bad_steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": target, "content": "hello\nworld\n"},
                "description": "create notes",
            },
            {
                "tool": "patch_file",
                "action": "replace",
                "params": {"file_path": target, "old": "invented", "new": "x"},
                "description": "patch notes",
            },
        ]
        result = orch.execute(bad_steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("EXECUTION FAILURE", provider.calls[0][0])
        self.assertEqual(notes_path.read_text(), "hello\neveryone\n")

    def test_no_evidence_returns_failure_with_replan_error(self):
        target = str(self.root / "notes.txt")
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "patch_file",
                "action": "replace",
                "params": {"file_path": str(target), "old": "nope", "new": "x"},
                "description": "patch",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("replan_error", result)
        self.assertIn("no request/evidence", result["replan_error"])

    def test_non_patch_failure_does_not_retry(self):
        target = self.root / "notes.txt"
        target.write_text("existing\n")
        provider = FakeProvider()
        orch = self._make_orch(provider)
        orch.last_request = "fix notes"
        orch.last_evidence = {"status": "ok", "request": "fix notes"}
        orch.last_impact = None
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {
                    "file_path": str(target),
                    "content": "clobber\n",
                    "rewrite": False,
                },
                "description": "refuse to clobber existing file",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(target.read_text(), "existing\n")


class TestBestEffortTests(OrchestratorTestCase):
    def test_soft_step_failure_keeps_changes_and_reports(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "notes.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "description": "create notes",
            },
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": 'echo "test failed" && exit 1'},
                "validate_after": True,
                "soft": True,
                "description": "run tests (best effort)",
            },
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(target.read_text(), "hello\n")
        self.assertFalse(result["trace"][1]["ok"])
        self.assertTrue(result["trace"][1]["soft_failure"])
        self.assertEqual(len(result["soft_failures"]), 1)
        self.assertEqual(result["testing"]["tested"], False)
        self.assertEqual(result["testing"]["status"], "tests_failed_after_changes")

    def test_soft_step_success_reports_tested(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        steps = [
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": 'echo "ok"'},
                "validate_after": True,
                "soft": True,
                "description": "run tests",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["testing"]["tested"], True)
        self.assertEqual(result["testing"]["status"], "tests_passed")

    def test_hard_step_still_rolls_back_after_soft_failure(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "notes.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "description": "create notes",
            },
            {
                "tool": "run_command",
                "action": "run",
                "params": {"command": 'echo "test failed" && exit 1'},
                "validate_after": True,
                "soft": True,
                "description": "run tests",
            },
            {
                "tool": "patch_file",
                "action": "replace",
                "params": {
                    "file_path": str(target),
                    "old": "not-present",
                    "new": "x",
                },
                "description": "hard failure step",
            },
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(target.exists())

    def test_no_soft_step_warns_user_about_custom_test(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "notes.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "description": "create notes",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        testing = result["testing"]
        self.assertEqual(testing["tested"], False)
        self.assertEqual(testing["status"], "no_test_framework")
        self.assertIn("custom test", testing["message"])


class TestUndoLog(OrchestratorTestCase):
    def test_rollback_restores_content(self):
        target = self.root / "f.txt"
        target.write_text("original")
        log = UndoLog(backup_root=str(self.root / "undo"))
        log.snapshot("write_file", "write", {"file_path": str(target)})
        target.write_text("changed")
        rollback = log.rollback()
        self.assertEqual(rollback["status"], "rolled_back")
        self.assertEqual(target.read_text(), "original")

    def test_rollback_removes_created_file(self):
        target = self.root / "new.txt"
        log = UndoLog(backup_root=str(self.root / "undo"))
        log.snapshot("write_file", "write", {"file_path": str(target)})
        target.write_text("new")
        log.rollback()
        self.assertFalse(target.exists())

    def test_snapshot_skips_read_tools(self):
        log = UndoLog(backup_root=str(self.root / "undo"))
        log.snapshot("read_file", "read", {"file_path": "x"})
        self.assertEqual(log.pending(), 0)


class TestOrchestratorTool(OrchestratorTestCase):
    def test_manual_has_actions(self):
        provider = FakeProvider()
        orch = self._make_tool(provider)
        self.assertIn("Actions:", orch.get_manual())
        self.assertEqual(orch.name, "orchestrator")

    def test_run_aborts_on_poor_discovery(self):
        (self.root / "decoy.txt").write_text("unrelated\n")
        provider = FakeProvider()
        orch = self._make_tool(provider)
        result = orch.dispatch("run", request="zzz nothing matches this at all")
        self.assertEqual(result["status"], "aborted")

    def test_run_requires_confirmation(self):
        target = self.root / "result.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                                "validate_after": True,
                                "expect": "done",
                                "description": "write result.txt",
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        result = orch.dispatch("run", request="create result.txt")
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertEqual(result["summary"][0]["tool"], "write_file")
        self.assertIn("confirm=true", result["message"])
        self.assertFalse(target.exists())

    def test_pending_reports_plan_after_confirmation_gate(self):
        target = self.root / "result.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        preview = orch.dispatch("run", request="create result.txt")
        self.assertEqual(preview["status"], "awaiting_confirmation")
        pending = orch.dispatch("pending")
        self.assertEqual(pending["status"], "pending_plan")
        self.assertEqual(pending["request"], "create result.txt")
        self.assertEqual(pending["steps"][0]["tool"], "write_file")

    def test_pending_empty_when_no_plan(self):
        provider = FakeProvider()
        orch = self._make_tool(provider)
        pending = orch.dispatch("pending")
        self.assertEqual(pending["status"], "no_pending_plan")

    def test_execute_resumes_pending_plan_without_steps(self):
        target = self.root / "result.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        orch.dispatch("run", request="create result.txt")
        result = orch.dispatch("execute", confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(target.read_text(), "done\n")

    def test_run_end_to_end(self):
        target = self.root / "result.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                                "validate_after": True,
                                "expect": "done",
                                "description": "write result.txt",
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        first = orch.dispatch("run", request="create result.txt")
        self.assertEqual(first["status"], "awaiting_confirmation")
        second = orch.dispatch("run", request="create result.txt", confirm=True)
        self.assertEqual(second["status"], "success")
        self.assertEqual(target.read_text(), "done\n")

    def test_run_reuses_pending_plan(self):
        target = self.root / "result.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        orch.dispatch("run", request="create result.txt")
        second = orch.dispatch("run", request="create result.txt", confirm=True)
        self.assertEqual(second["status"], "success")
        self.assertEqual(target.read_text(), "done\n")

    def test_undo_after_success(self):
        target = self.root / "f.txt"
        target.write_text("before")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "after\n",
                                },
                                "rewrite": True,
                            }
                        ]
                    }
                )
            ]
        )
        orch = self._make_tool(provider)
        result = orch.dispatch("run", request="edit f.txt", confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(target.read_text(), "after\n")
        undo = orch.dispatch("undo")
        self.assertEqual(undo["status"], "rolled_back")
        self.assertEqual(target.read_text(), "before")

    def test_awaiting_confirmation_includes_impact(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        result = orch._awaiting_confirmation(
            [
                {
                    "tool": "patch_file",
                    "action": "replace",
                    "params": {"file_path": "x.py"},
                    "description": "patch x.py",
                }
            ],
            impact={
                "targets": [
                    {
                        "path": "/tmp/x.py",
                        "change_mode": "modify",
                        "recommended_tool": "patch_file",
                        "risks": ["must keep contract"],
                    }
                ]
            },
        )
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertIn("Impact analysis", result["impact"])
        self.assertIn("mode=modify", result["impact"])

    def test_unknown_action(self):
        provider = FakeProvider()
        orch = self._make_tool(provider)
        result = orch.dispatch("nope")
        self.assertEqual(result["error"], "unknown_action")

    def test_registered_then_removed(self):
        provider = FakeProvider()
        orch = self._make_tool(provider)
        self.registry.register(orch.name, orch)
        self.assertIn("orchestrator", self.registry.list_tools())
        self.assertTrue(self.registry.unregister("orchestrator"))
        self.assertNotIn("orchestrator", self.registry.list_tools())
        self.assertIn("write_file", self.registry.list_tools())


class TestMemoryIndependence(unittest.TestCase):
    """The orchestrator never shares the agent's memory: step results stay in
    an independent, compressor-free instance and are reset per invocation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        os.environ["ORCH_CONFIRM_MODE"] = "agent"
        self.addCleanup(os.environ.pop, "ORCH_CONFIRM_MODE", None)

    def test_agent_memory_untouched_and_local_memory_used(self):
        from src.memory import Memory

        target = self.root / "f.txt"
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "x",
                                },
                            }
                        ]
                    }
                )
            ]
        )
        agent_memory = Memory()
        orch = Orchestrator(build_default_registry(), provider, agent_memory)
        result = orch.run(request="create f.txt", confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(agent_memory), 0)
        # Local (independent) memory recorded the executed step.
        self.assertEqual(len(orch.memory), 1)
        self.assertIsNone(orch.memory.compressor)

    def test_local_memory_reset_between_runs(self):
        target = self.root / "f.txt"
        plan = json.dumps(
            {
                "steps": [
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {"file_path": str(target), "content": "x"},
                    }
                ]
            }
        )
        provider = FakeProvider([plan])
        orch = Orchestrator(build_default_registry(), provider)
        first = orch.run(request="create f.txt", confirm=True)
        second = orch.run(request="create f.txt", confirm=True)
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        self.assertEqual(len(orch.memory), 1)


class TestInteractiveConfirmation(unittest.TestCase):
    """With a TTY stdin the orchestrator asks for approval itself via input()
    and executes or cancels in the SAME call — nothing round-trips through
    the main agent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        os.environ.pop("ORCH_CONFIRM_MODE", None)
        self.addCleanup(os.environ.pop, "ORCH_CONFIRM_MODE", None)
        # Silence the plan summary the gate prints to stdout.
        self._stdout_patcher = patch("sys.stdout")
        self._stdout_patcher.start()
        self.addCleanup(self._stdout_patcher.stop)

    def _plan(self, target: Path) -> FakeProvider:
        return FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(target),
                                    "content": "done\n",
                                },
                            }
                        ]
                    }
                )
            ]
        )

    def test_approval_executes_inline(self):
        target = self.root / "result.txt"
        orch = Orchestrator(
            build_default_registry(), self._plan(target), root=str(self.root)
        )
        with patch("sys.stdin") as fake_stdin, patch(
            "builtins.input", return_value="s"
        ) as fake_input:
            fake_stdin.isatty.return_value = True
            result = orch.run(request="create result.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(target.read_text(), "done\n")
        fake_input.assert_called_once()

    def test_rejection_cancels_and_discards(self):
        target = self.root / "result.txt"
        orch = Orchestrator(
            build_default_registry(), self._plan(target), root=str(self.root)
        )
        with patch("sys.stdin") as fake_stdin, patch(
            "builtins.input", return_value="n"
        ):
            fake_stdin.isatty.return_value = True
            result = orch.run(request="create result.txt")
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(target.exists())
        # A rejected plan leaves no pending state behind.
        self.assertEqual(orch.pending()["status"], "no_pending_plan")

    def test_eof_counts_as_rejection(self):
        target = self.root / "result.txt"
        orch = Orchestrator(
            build_default_registry(), self._plan(target), root=str(self.root)
        )
        with patch("sys.stdin") as fake_stdin, patch(
            "builtins.input", side_effect=EOFError
        ):
            fake_stdin.isatty.return_value = True
            result = orch.run(request="create result.txt")
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(target.exists())

    def test_non_tty_falls_back_to_legacy_preview(self):
        target = self.root / "result.txt"
        orch = Orchestrator(
            build_default_registry(), self._plan(target), root=str(self.root)
        )
        with patch("sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            result = orch.run(request="create result.txt")
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertFalse(target.exists())


class TestPatchGeneration(unittest.TestCase):
    """patch_file steps carry an instruction; the concrete payload is
    generated at execution time from the CURRENT disk content by the
    dedicated generator (atomic steps, no cross-step anchor copying)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.target = self.root / "f.txt"

    def _orch(self, provider, **kwargs) -> Orchestrator:
        return Orchestrator(
            build_default_registry(), provider, root=str(self.root), **kwargs
        )

    def _instruction_step(self) -> dict:
        return {
            "tool": "patch_file",
            "action": "replace",
            "params": {
                "file_path": str(self.target),
                "instruction": "replace hello with goodbye",
            },
            "description": "edit f.txt",
        }

    def test_instruction_generates_and_applies_patch(self):
        self.target.write_text("hello world\nsecond line\n")
        provider = FakeProvider(
            [
                json.dumps({"steps": [self._instruction_step()]}),
                json.dumps({"mode": "replace", "old": "hello", "new": "goodbye"}),
            ]
        )
        orch = self._orch(provider)
        result = orch.execute([self._instruction_step()], confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.target.read_text(), "goodbye world\nsecond line\n")
        trace_entry = result["trace"][0]
        self.assertTrue(trace_entry["patch_generated"])
        # The orchestration-only hint never leaks into the dispatched params.
        self.assertNotIn("instruction", trace_entry["result"])

    def test_generator_sees_only_instruction_and_content(self):
        import re

        self.target.write_text("hello world\n")
        provider = FakeProvider(
            [json.dumps({"mode": "replace", "old": "hello", "new": "goodbye"})]
        )
        generator = PatchGenerator(provider)
        generated = generator.generate(
            str(self.target),
            self.target.read_text(),
            "replace hello with goodbye",
        )
        self.assertEqual(generated["status"], "ok")
        user_prompt, config = provider.calls[0]
        # Minimal context: no tool manual, no plan JSON mixed in.
        self.assertNotIn("Available tools", user_prompt)
        self.assertNotIn('"steps"', user_prompt)
        self.assertIn("replace hello with goodbye", user_prompt)
        self.assertIn("hello world", user_prompt)
        self.assertIn("VERBATIM", config.upper())

    def test_invalid_anchor_retries_then_fails_with_rollback(self):
        self.target.write_text("hello world\n")
        bad = json.dumps({"mode": "replace", "old": "nope", "new": "x"})
        provider = FakeProvider([bad, bad])
        orch = self._orch(provider)
        result = orch.execute([self._instruction_step()], confirm=True)
        self.assertEqual(result["status"], "failed")
        # Two generation attempts (initial + one retry).
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("not found", result["trace"][0]["result"]["error"]["message"])
        self.assertEqual(self.target.read_text(), "hello world\n")

    def test_generation_failure_message_included_in_trace(self):
        self.target.write_text("hello world\n")
        provider = FakeProvider(
            [
                json.dumps({"steps": [self._instruction_step()]}),
                "not json at all",
                "still not json",
            ]
        )
        orch = self._orch(provider)
        result = orch.execute([self._instruction_step()], confirm=True)
        self.assertEqual(result["status"], "failed")
        message = result["trace"][0]["result"]["error"]["message"]
        self.assertIn("invalid patch", message)

    def test_explicit_anchor_steps_skip_generator(self):
        self.target.write_text("hello world\n")
        step = {
            "tool": "patch_file",
            "action": "replace",
            "params": {
                "file_path": str(self.target),
                "old": "hello",
                "new": "goodbye",
            },
        }
        provider = FakeProvider()
        orch = self._orch(provider)
        result = orch.execute([step], confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["trace"][0]["patch_generated"])
        self.assertEqual(len(provider.calls), 0)

    def test_unreadable_target_fails_cleanly(self):
        missing = self.root / "missing.txt"
        step = {
            "tool": "patch_file",
            "action": "replace",
            "params": {
                "file_path": str(missing),
                "instruction": "anything",
            },
        }
        provider = FakeProvider([json.dumps({"steps": [step]}), json.dumps({})])
        orch = self._orch(provider)
        result = orch.execute([step], confirm=True)
        self.assertEqual(result["status"], "failed")
        message = result["trace"][0]["result"]["error"]["message"]
        self.assertIn("could not read", message)

    def test_planner_prompt_forbids_hand_written_anchors(self):
        provider = FakeProvider()
        prompt = self._orch(provider).planner.build_prompt(
            "request", {"candidates": []}
        )
        self.assertIn("instruction", prompt)
        self.assertIn("ATOMIC", prompt)
        self.assertIn("NEVER invent 'old', 'new' or 'diff'", prompt)

    def test_plan_accepts_instruction_step_without_prior_read(self):
        self.target.write_text("hello world\n")
        plan = json.dumps({"steps": [self._instruction_step()]})
        provider = FakeProvider([plan])
        orch = self._orch(provider)
        result = orch.plan("edit f.txt replacing hello with goodbye")
        self.assertEqual(result["status"], "ok")


class TestPatchGeneratorValidation(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.generator = PatchGenerator(self.provider)

    def test_replace_requires_unique_anchor(self):
        error, params = PatchGenerator._validate(
            {"mode": "replace", "old": "a", "new": "b"}, "a\na\n"
        )
        self.assertIn("2 times", error)
        self.assertEqual(params, {})

    def test_replace_all_allows_multiple(self):
        error, params = PatchGenerator._validate(
            {"mode": "replace", "old": "a", "new": "b", "replace_all": True},
            "a\na\n",
        )
        self.assertIsNone(error)
        self.assertEqual(params, {"old": "a", "new": "b", "replace_all": True})

    def test_apply_validates_hunks_against_content(self):
        diff = "--- f\n+++ f\n@@ -1,1 +1,1 @@\n-hello\n+goodbye\n"
        ok_error, ok_params = PatchGenerator._validate(
            {"mode": "apply", "diff": diff}, "hello\n"
        )
        self.assertIsNone(ok_error)
        self.assertEqual(ok_params["diff"], diff)
        bad_error, _ = PatchGenerator._validate(
            {"mode": "apply", "diff": diff}, "different content\n"
        )
        self.assertIn("not found", bad_error)

    def test_mode_inferred_from_payload_shape(self):
        error, params = PatchGenerator._validate({"old": "a", "new": "b"}, "a\n")
        self.assertIsNone(error)
        self.assertEqual(params, {"old": "a", "new": "b"})


class TestPreAssessment(unittest.TestCase):
    def test_simple_request_stays_monolithic(self):
        assessment = PreAssessor(threshold=3).assess("corrija o bug na funcao parse")
        self.assertFalse(assessment["needs_split"])

    def test_enumerated_request_needs_split(self):
        request = "1. criar models\n2. criar rotas\n3. escrever testes"
        assessment = PreAssessor(threshold=3).assess(request)
        self.assertTrue(assessment["needs_split"])
        self.assertEqual(assessment["signals"]["enumeration_items"], 3)

    def test_env_threshold_override(self):
        with patch.dict(os.environ, {"ORCH_SPLIT_THRESHOLD": "1"}):
            assessor = PreAssessor()
            self.assertEqual(assessor.threshold, 1)
            # a single connector is enough at threshold 1
            self.assertTrue(
                assessor.assess("mude o titulo e depois mude o rodape")["needs_split"]
            )

    def test_explicit_constructor_threshold_wins_over_env(self):
        with patch.dict(os.environ, {"ORCH_SPLIT_THRESHOLD": "1"}):
            assessor = PreAssessor(threshold=5)
            self.assertEqual(assessor.threshold, 5)

    def test_decomposer_uses_llm_output(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "parts": [
                            {"id": "X", "request": "parte A", "paths": ["src/a.py"]},
                            {"id": "Y", "request": "parte B"},
                        ]
                    }
                )
            ]
        )
        result = Decomposer(provider).decompose("pedido composto")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["strategy"], "llm")
        self.assertEqual([p["id"] for p in result["parts"]], ["PART-001", "PART-002"])
        self.assertEqual(result["parts"][0]["paths"], ["src/a.py"])
        self.assertEqual(result["parts"][1]["paths"], [])

    def test_decomposer_falls_back_to_heuristics(self):
        class Down:
            def infer(self, prompt, config):
                raise RuntimeError("provider down")

        request = "1. criar models\n2. criar rotas\n3. escrever testes"
        result = Decomposer(Down()).decompose(request)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["strategy"], "heuristic_fallback")
        self.assertEqual(len(result["parts"]), 3)
        self.assertIn("models", result["parts"][0]["request"])

    def test_decomposer_single_part_means_no_split(self):
        provider = FakeProvider([json.dumps({"parts": [{"request": "tudo junto"}]})])
        result = Decomposer(provider).decompose("pedido simples")
        self.assertEqual(result["status"], "single")

    def test_decomposer_merges_excess_parts(self):
        parts = [{"request": f"item {i}"} for i in range(9)]
        provider = FakeProvider([json.dumps({"parts": parts})])
        result = Decomposer(provider, max_parts=4).decompose("pedido gigante")
        self.assertEqual(len(result["parts"]), 4)
        self.assertIn("item 8", result["parts"][-1]["request"])


class TestUndoLogCheckpoints(unittest.TestCase):
    def test_rollback_to_keeps_prior_entries(self):
        log = UndoLog()
        # inert entries (restore of a non-existing path is a no-op delete)
        entry = {
            "tool": "write_file",
            "action": "write",
            "kind": "path",
            "state": {
                "type": "none",
                "path": Path(log.backup_root) / "x",
                "existed": False,
            },
        }
        log._entries.append(dict(entry))
        mark = log.checkpoint()
        self.assertEqual(mark, 1)
        log._entries.append(dict(entry, tool="patch_file"))
        result = log.rollback_to(mark)
        self.assertEqual(result["restored"], 1)
        self.assertEqual(len(log._entries), 1)
        self.assertEqual(log._entries[0]["tool"], "write_file")
        # clamping beyond bounds is safe (nothing to restore)
        self.assertEqual(log.rollback_to(99)["restored"], 0)
        self.assertEqual(len(log._entries), 1)


class TestScoreRequest(unittest.TestCase):
    def test_connectors_and_verbs_count_points(self):
        request = (
            "Crie uma API de produtos.\n"
            "Adicione autenticacao JWT.\n"
            "Tambem configure o banco de dados e escreva testes."
        )
        score = score_request(request)
        self.assertGreaterEqual(score["points"], 3)

    def test_word_boundary_avoids_prefix_double_count(self):
        score = score_request("criar models e criar rotas")
        # 'cria' must NOT also match inside 'criar'
        self.assertNotIn("cria", score["signals"]["change_verbs"])


class TestDecompositionFlow(OrchestratorTestCase):
    """End-to-end multi-part runs in legacy (agent relay) mode."""

    def _parted_orch(self, responses):
        with patch.dict(os.environ, {"ORCH_SPLIT_THRESHOLD": "1"}):
            orch = Orchestrator(
                self.registry, FakeProvider(responses), root=str(self.root)
            )
        return orch

    def _compound(self):
        return (
            f"Crie o arquivo {self.root / 'report.txt'} com conteudo report.\n"
            f"Tambem crie o arquivo {self.root / 'notes.txt'} com conteudo notes."
        )

    def _decomposition(self):
        return json.dumps(
            {
                "parts": [
                    {
                        "id": "P1",
                        "request": "create report",
                        "rationale": "",
                        "paths": [str(self.root / "report.txt")],
                    },
                    {
                        "id": "P2",
                        "request": "create notes",
                        "rationale": "",
                        "paths": [str(self.root / "notes.txt")],
                    },
                ]
            }
        )

    def _plan_for(self, path, content):
        return json.dumps(
            {
                "steps": [
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {"file_path": str(path), "content": content},
                        "description": f"write {Path(path).name}",
                    }
                ]
            }
        )

    def test_simple_request_below_threshold_skips_decomposition(self):
        (self.root / "target.txt").write_text("hello\n")
        plan = json.dumps(
            {
                "steps": [
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": str(self.root / "target.txt")},
                        "description": "read",
                    }
                ]
            }
        )
        orch = self._parted_orch([plan])
        result = orch.run("leia o arquivo target.txt")
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertNotIn("part", result)
        self.assertFalse(orch.split_active)

    def test_compound_request_splits_and_executes_per_part(self):
        report = self.root / "report.txt"
        notes = self.root / "notes.txt"
        orch = self._parted_orch(
            [
                self._decomposition(),
                self._plan_for(report, "report"),
                self._plan_for(notes, "notes"),
            ]
        )
        first = orch.run(self._compound())
        self.assertEqual(first["status"], "awaiting_confirmation")
        self.assertEqual(first["part"]["index"], 1)
        self.assertEqual(first["part"]["total"], 2)
        states = {p["id"]: p["state"] for p in first["parts_overview"]}
        self.assertEqual(states, {"PART-001": "current", "PART-002": "pending"})
        self.assertFalse(report.exists())

        second = orch.execute(confirm=True)
        self.assertEqual(second["status"], "awaiting_confirmation")
        self.assertEqual(second["part"]["index"], 2)
        self.assertTrue(report.exists())
        self.assertFalse(notes.exists())

        final = orch.execute(confirm=True)
        self.assertEqual(final["status"], "success")
        self.assertTrue(final["decomposed"])
        self.assertEqual(
            [p["id"] for p in final["parts_executed"]],
            ["PART-001", "PART-002"],
        )
        self.assertEqual(report.read_text(), "report")
        self.assertEqual(notes.read_text(), "notes")
        self.assertFalse(orch.split_active)
        self.assertIsNone(orch.last_parts)

    def test_pending_reports_part_metadata(self):
        orch = self._parted_orch(
            [
                self._decomposition(),
                self._plan_for(self.root / "report.txt", "r"),
                self._plan_for(self.root / "notes.txt", "n"),
            ]
        )
        orch.run(self._compound())
        pending = orch.pending()
        self.assertEqual(pending["status"], "pending_plan")
        self.assertEqual(pending["part"]["index"], 1)
        self.assertEqual(len(pending["parts_overview"]), 2)

    def test_failure_keeps_completed_parts_and_rolls_back_only_failed(self):
        report = self.root / "report.txt"
        doomed = json.dumps(
            {
                "steps": [
                    {
                        "tool": "run_command",
                        "action": "run",
                        "params": {"command": "exit 3"},
                        "description": "doomed",
                    }
                ]
            }
        )
        orch = self._parted_orch(
            [
                self._decomposition(),
                self._plan_for(report, "report"),
                doomed,
            ]
        )
        orch.run(self._compound())
        orch.execute(confirm=True)  # part 1 ok
        stopped = orch.execute(confirm=True)  # part 2 fails
        self.assertEqual(stopped["status"], "decomposition_stopped")
        self.assertIn("execution failed", stopped["reason"])
        self.assertEqual([p["id"] for p in stopped["completed_parts"]], ["PART-001"])
        self.assertEqual([p["id"] for p in stopped["remaining_parts"]], ["PART-002"])
        # part 1 stays applied; split state remains resumable
        self.assertTrue(report.exists())
        self.assertTrue(orch.split_active)
        self.assertTrue(orch.part_failed)
        rollback = (stopped.get("execution") or {}).get("rollback") or {}
        self.assertEqual(rollback.get("restored"), 0)

    def test_resume_after_failure_replans_with_feedback(self):
        report = self.root / "report.txt"
        a_txt = self.root / "alpha.txt"
        a_txt.write_text("alpha\n")
        doomed = json.dumps(
            {
                "steps": [
                    {
                        "tool": "run_command",
                        "action": "run",
                        "params": {"command": "exit 3"},
                        "description": "doomed",
                    }
                ]
            }
        )
        good = json.dumps(
            {
                "steps": [
                    {
                        "tool": "read_file",
                        "action": "read",
                        "params": {"file_path": str(a_txt)},
                        "description": "read alpha",
                    },
                    {
                        "tool": "patch_file",
                        "action": "replace",
                        "params": {
                            "file_path": str(a_txt),
                            "old": "alpha",
                            "new": "alpha DONE",
                        },
                        "description": "good patch",
                    },
                ]
            }
        )
        decomposition = json.dumps(
            {
                "parts": [
                    {"id": "PA", "request": "create report", "paths": [str(report)]},
                    {"id": "PB", "request": "adjust alpha text", "paths": [str(a_txt)]},
                ]
            }
        )
        orch = self._parted_orch(
            [decomposition, self._plan_for(report, "report"), doomed, good]
        )
        orch.run(self._compound() + " e ajuste o texto")
        orch.execute(confirm=True)
        stopped = orch.execute(confirm=True)
        self.assertEqual(stopped["status"], "decomposition_stopped")

        resumed = orch.run(None)  # resume: replans part 2
        self.assertEqual(resumed["status"], "awaiting_confirmation")
        self.assertEqual(resumed["part"]["index"], 2)
        planner_prompt = orch.provider.calls[-1][0]
        self.assertIn("EXECUTION FAILURE", planner_prompt)

        final = orch.execute(confirm=True)
        self.assertEqual(final["status"], "success")
        self.assertEqual(a_txt.read_text(), "alpha DONE\n")

    def test_explicit_paths_disable_splitting(self):
        (self.root / "solo.txt").write_text("x\n")
        new_path = self.root / "brand-new.txt"
        plan = json.dumps(
            {
                "steps": [
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {
                            "file_path": str(new_path),
                            "content": "fresh",
                        },
                        "description": "create",
                    }
                ]
            }
        )
        orch = self._parted_orch([plan])
        compound = (
            f"Crie {new_path} e tambem ajuste tudo no solo.txt com outras "
            "coisas pendentes"
        )
        result = orch.run(compound, paths=[str(new_path)])
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertNotIn("part", result)
        self.assertFalse(orch.split_active)

    def test_abort_clears_part_state(self):
        orch = self._parted_orch(
            [
                self._decomposition(),
                self._plan_for(self.root / "report.txt", "r"),
                self._plan_for(self.root / "notes.txt", "n"),
            ]
        )
        orch.run(self._compound())
        aborted = orch.abort()
        self.assertEqual(aborted["status"], "aborted")
        self.assertFalse(orch.split_active)
        self.assertIsNone(orch.last_parts)
        self.assertEqual(orch.pending()["status"], "no_pending_plan")


if __name__ == "__main__":
    unittest.main()
