import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.orchestrator import create_orchestrator_tool
from src.orchestrator.impact import TOOL_CONTRACT, ImpactAssessor
from src.orchestrator.orchestrator import Orchestrator
from src.tools.registry import build_default_registry

REPO = Path(__file__).resolve().parents[1]


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


class ImpactTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.registry = build_default_registry()

    def _make_temp_tool(self, name: str, broken: bool = False):
        pkg = self.root / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        tool_path = pkg / "tool.py"
        if broken:
            tool_path.write_text("broken = 1\n")
        else:
            tool_path.write_text(
                "def get_manual():\n"
                "    return 'manual'\n\n"
                "def dispatch(action, **params):\n"
                "    return {'status': 'success'}\n"
            )
        sys.path.insert(0, str(self.root))
        module = importlib.import_module(f"{name}.tool")

        def _cleanup():
            sys.modules.pop(f"{name}.tool", None)
            sys.modules.pop(name, None)
            try:
                sys.path.remove(str(self.root))
            except ValueError:
                pass

        self.addCleanup(_cleanup)
        return tool_path, module


class TestProfile(ImpactTestCase):
    def test_new_target(self):
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "notes.txt"))
        self.assertFalse(prof["exists"])
        self.assertEqual(prof["change_mode"], "new")
        self.assertEqual(prof["recommended_tool"], "write_file")

    def test_existing_non_tool_target(self):
        (self.root / "notes.txt").write_text("hello\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "notes.txt"))
        self.assertTrue(prof["exists"])
        self.assertEqual(prof["change_mode"], "modify")
        self.assertEqual(prof["recommended_tool"], "patch_file")
        self.assertFalse(prof["is_registered_tool"])

    def test_registered_tool_profile(self):
        assessor = ImpactAssessor(self.registry, root=str(REPO))
        prof = assessor.profile(str(REPO / "src/tools/list_dir.py"))
        self.assertTrue(prof["is_registered_tool"])
        self.assertTrue(prof["contract_preserved"])
        self.assertEqual(prof["change_mode"], "modify")
        self.assertIn("def get_manual", prof.get("top_level", []))
        self.assertIn("def dispatch", prof.get("top_level", []))
        self.assertTrue(prof["test_files"])
        self.assertIn(" -m unittest ", prof["test_command"])


class TestImpactAnalysis(ImpactTestCase):
    def test_risks_include_references_and_tests(self):
        assessor = ImpactAssessor(self.registry, root=str(REPO))
        prof = assessor.profile(str(REPO / "src/tools/list_dir.py"))
        risks = assessor.risks(prof)
        self.assertTrue(any("imported by" in r for r in risks))
        self.assertTrue(any("test file" in r for r in risks))

    def test_assess_returns_targets(self):
        assessor = ImpactAssessor(self.registry, root=str(REPO))
        result = assessor.assess(
            "add exclude to list_dir",
            [{"path": str(REPO / "src/tools/list_dir.py"), "type": "file"}],
        )
        self.assertEqual(len(result["targets"]), 1)
        target = result["targets"][0]
        self.assertTrue(target["is_registered_tool"])
        self.assertIn("risks", target)

    def test_assess_tool_action(self):
        tool = create_orchestrator_tool(self.registry, FakeProvider(), root=str(REPO))
        result = tool.dispatch(
            "assess", request="add exclude to list_dir", terms=["list_dir"]
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["targets"])


class TestPlanGuard(ImpactTestCase):
    def test_rejects_write_over_registered_tool(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(REPO / "src/tools/list_dir.py"),
                                    "content": "x = 1\n",
                                },
                                "validate_after": True,
                                "description": "overwrite list_dir",
                            }
                        ]
                    }
                )
            ]
        )
        orch = Orchestrator(self.registry, provider, root=str(REPO))
        plan = orch.plan("add exclude to list_dir", terms=["list_dir"])
        self.assertEqual(plan["status"], "error")
        self.assertIn("rewrite", plan["message"])

    def test_run_returns_error_on_clobber_plan(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "write_file",
                                "action": "write",
                                "params": {
                                    "file_path": str(REPO / "src/tools/list_dir.py"),
                                    "content": "x = 1\n",
                                },
                                "validate_after": True,
                                "description": "overwrite list_dir",
                            }
                        ]
                    }
                )
            ]
        )
        orch = Orchestrator(self.registry, provider, root=str(REPO))
        result = orch.run("add exclude to list_dir", terms=["list_dir"])
        self.assertEqual(result["status"], "error")

    def test_accepts_new_file_write(self):
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(self.root / "new.txt"), "content": "x\n"},
            }
        ]
        self.assertEqual(assessor.check_plan_steps(steps), [])

    def test_accepts_rewrite_flag(self):
        assessor = ImpactAssessor(self.registry, root=str(REPO))
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {
                    "file_path": str(REPO / "src/tools/list_dir.py"),
                    "content": "x = 1\n",
                },
                "rewrite": True,
            }
        ]
        self.assertEqual(assessor.check_plan_steps(steps), [])

    def test_awaiting_confirmation_includes_impact(self):
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "steps": [
                            {
                                "tool": "patch_file",
                                "action": "replace",
                                "params": {
                                    "file_path": str(REPO / "src/tools/list_dir.py"),
                                    "old": "def handle_list_dir(",
                                    "new": "def handle_list_dir(",
                                },
                                "validate_after": True,
                                "expect": "handle_list_dir",
                                "description": "patch list_dir",
                            }
                        ]
                    }
                )
            ]
        )
        orch = Orchestrator(self.registry, provider, root=str(REPO))
        result = orch.run("add exclude to list_dir", terms=["list_dir"])
        self.assertEqual(result["status"], "awaiting_confirmation")
        self.assertIn("Impact analysis", result.get("impact", ""))
        self.assertIn("mode=modify", result["impact"])
        self.assertIn("recommended patch_file", result["impact"])


class TestLanguageAwareProfile(ImpactTestCase):
    def test_typescript_profile_maps_sibling_test(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "calc.ts").write_text(
            "export const add = (a, b) => a + b;\n"
        )
        (self.root / "src" / "calc.test.ts").write_text(
            "import { add } from './calc';\n"
        )
        (self.root / "package.json").write_text(
            '{"scripts": {"test": "jest"}, "dependencies": {"jest": "29"}}\n'
        )
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "src/calc.ts"))
        self.assertEqual(prof["language"], "typescript")
        self.assertTrue(prof["test_files"])
        self.assertIn("jest", prof["test_command"])
        self.assertTrue(prof["test_framework_configured"])
        self.assertEqual(prof["test_success_marker"], "PASS")

    def test_java_suggestion_when_framework_configured_but_uncovered(self):
        (self.root / "src" / "main" / "java").mkdir(parents=True)
        (self.root / "src/main/java/Counter.java").write_text("class Counter {}\n")
        (self.root / "pom.xml").write_text("<project/>\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "src/main/java/Counter.java"))
        self.assertEqual(prof["language"], "java")
        self.assertEqual(prof["test_files"], [])
        self.assertTrue(prof["test_framework_configured"])
        self.assertIn("CounterTest", prof["test_suggestion"]["new_test_path"])
        self.assertEqual(prof["test_suggestion"]["success_marker"], "BUILD SUCCESS")

    def test_c_recommendation_when_no_toolchain(self):
        (self.root / "stack.c").write_text("int push() { return 0; }\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "stack.c"))
        self.assertEqual(prof["language"], "c")
        self.assertFalse(prof["test_framework_configured"])
        self.assertNotIn("test_suggestion", prof)
        self.assertIn("configure", prof["test_recommendation"])

    def test_unknown_language_has_no_tests_and_recommendation(self):
        (self.root / "notes.txt").write_text("hello\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "notes.txt"))
        self.assertIsNone(prof["language"])
        self.assertEqual(prof["test_files"], [])
        self.assertFalse(prof["test_framework_configured"])
        self.assertIn("test", prof["test_recommendation"])

    def test_references_search_language_family(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "calc.ts").write_text(
            "export const add = (a, b) => a + b;\n"
        )
        (self.root / "src" / "use.ts").write_text("import { add } from './calc';\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        refs = assessor._referenced_by(self.root / "src/calc.ts")
        self.assertTrue(any("use.ts" in r for r in refs))

    def test_format_impact_shows_language_and_tiers(self):
        (self.root / "stack.c").write_text("int push() { return 0; }\n")
        assessor = ImpactAssessor(self.registry, root=str(self.root))
        prof = assessor.profile(str(self.root / "stack.c"))
        prof["risks"] = assessor.risks(prof)
        rendered = Orchestrator._format_impact({"targets": [prof]})
        self.assertIn("language: c", rendered)
        self.assertIn("no test setup", rendered)


class TestPlannerPromptRules(ImpactTestCase):
    def test_prompt_is_language_aware(self):
        from src.orchestrator.planner import Planner

        provider = FakeProvider()
        planner = Planner(provider, self.registry)
        prompt = planner.build_prompt(
            "change calc.ts",
            {"impact": {"targets": [{"language": "typescript"}]}},
        )
        self.assertIn("ANY language", prompt)
        self.assertIn("typescript", prompt)
        self.assertIn("test_suggestion", prompt)
        self.assertIn("test_recommendation", prompt)


class TestExecutorDeepValidation(ImpactTestCase):
    def test_broken_rewrite_rolls_back(self):
        tool_path, module = self._make_temp_tool("pkg_rollback")
        registry = build_default_registry()
        registry.register("pkg_rollback_tool", module)
        orch = Orchestrator(registry, FakeProvider(), root=str(self.root))
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(tool_path), "content": "broken = 1\n"},
                "rewrite": True,
                "validate_after": True,
                "description": "rewrite tool module without contract",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rollback"]["status"], "rolled_back")
        self.assertIn("get_manual", tool_path.read_text())

    def test_contract_preserving_rewrite_succeeds(self):
        tool_path, module = self._make_temp_tool("pkg_keep")
        registry = build_default_registry()
        registry.register("pkg_keep_tool", module)
        orch = Orchestrator(registry, FakeProvider(), root=str(self.root))
        content = (
            "MANUAL = 'manual'\n"
            "SPEC = 'spec'\n\n"
            "def get_manual():\n"
            "    return 'new manual'\n\n"
            "def dispatch(action, **params):\n"
            "    return {'status': 'success'}\n"
        )
        steps = [
            {
                "tool": "write_file",
                "action": "write",
                "params": {"file_path": str(tool_path), "content": content},
                "rewrite": True,
                "validate_after": True,
                "expect": "get_manual",
                "description": "rewrite tool module keeping contract",
            }
        ]
        result = orch.execute(steps, confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(tool_path.read_text(), content)


if __name__ == "__main__":
    unittest.main()
