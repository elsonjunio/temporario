import json
import tempfile
import unittest
from pathlib import Path

from src.orchestrator import create_orchestrator_tool
from src.orchestrator.discovery import Discovery
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.undo import UndoLog
from src.tools.registry import build_default_registry


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


class OrchestratorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.registry = build_default_registry()

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


class TestDiscovery(OrchestratorTestCase):
    def test_finds_target(self):
        (self.root / "notes.txt").write_text("foo bar\n")
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("change something in notes.txt")
        self.assertEqual(result["status"], "ok")
        paths = [c["path"] for c in result["candidates"]]
        self.assertTrue(any("notes.txt" in p for p in paths))

    def test_poor_when_no_match(self):
        discovery = Discovery(self.registry, root=str(self.root))
        result = discovery.discover("xyzzy zorp unknown")
        self.assertEqual(result["status"], "poor")

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

    def test_validation_expect_failure_rolls_back(self):
        provider = FakeProvider()
        orch = self._make_orch(provider)
        target = self.root / "out.txt"
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(target), "content": "hello\n"},
                "validate_after": True,
                "expect": "WRONG",
                "description": "create out.txt",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rollback"]["status"], "rolled_back")
        self.assertFalse(target.exists())

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


if __name__ == "__main__":
    unittest.main()
