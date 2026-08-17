"""Tests for the Discovery Agent subagent: private read-only registry, the
``code`` symbol toolset, and the investigation loop with budget + telemetry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.discovery import (
    DiscoveryAgent,
    build_code_spec,
    build_discovery_registry,
    create_discovery_tool,
)
from src.discovery import symbols


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        if self.responses:
            return self.responses.pop(0)
        return "DISCOVERY RESULT\n\nTASK\nok\n\nSUMMARY\nok"


FINAL_REPORT = (
    "DISCOVERY RESULT\n\n"
    "TASK\nfind the auth flow\n\n"
    "SUMMARY\nAuthentication is in src/auth.\n\n"
    "RELEVANT FILES\n"
    "1. src/auth/service.py\n"
    "   Purpose: login logic\n"
    "   Relevant symbols: AuthenticationService.authenticate()\n"
    "   Relevance: entry point\n"
    "   Important details: called by src/api/auth.py:42\n\n"
    "RELATIONSHIPS\nsrc/api/auth.py -> src/auth/service.py\n\n"
    "IMPLEMENTATION AREA\nsrc/auth/service.py\n\n"
    "EVIDENCE\nfind_references found call sites.\n\n"
    "RISKS\nnone.\n\n"
    "NOT INVESTIGATED\ntests\n\n"
    "CONFIDENCE\nmedium"
)


def _make_project(root: Path) -> None:
    (root / "src" / "auth").mkdir(parents=True)
    (root / "src" / "api").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "auth" / "__init__.py").write_text("")
    (root / "src" / "auth" / "service.py").write_text(
        '"""Authentication service."""\n\n'
        "class AuthenticationService:\n"
        "    def authenticate(self, email, password):\n"
        '        return email == "admin@x.com"\n'
    )
    (root / "src" / "api" / "__init__.py").write_text("")
    (root / "src" / "api" / "auth.py").write_text(
        "from src.auth.service import AuthenticationService\n"
        "\n"
        "def login():\n"
        "    service = AuthenticationService()\n"
        '    return service.authenticate("a", "b")\n'
    )
    (root / "tests" / "test_auth.py").write_text(
        "from src.auth.service import AuthenticationService\n"
        "\n"
        "def test_authenticate():\n"
        '    assert AuthenticationService().authenticate("a", "b")\n'
    )


class TestDiscoveryRegistry(unittest.TestCase):
    def test_only_read_only_tools(self):
        registry = build_discovery_registry()
        tools = set(registry.list_tools())
        self.assertIn("list_dir", tools)
        self.assertIn("search_files", tools)
        self.assertIn("grep_files", tools)
        self.assertIn("read_file", tools)
        self.assertIn("code", tools)
        self.assertTrue(
            tools
            & {
                "list_files",
                "ls",
                "read",
                "cat",
                "search",
                "grep",
                "find_symbol",
                "find_definition",
                "find_references",
                "find_imports",
                "find_importers",
                "inspect",
                "inspect_file",
            }
        )
        self.assertFalse(
            tools
            & {"write_file", "patch_file", "delete_file", "move_file", "run_command"}
        )

    def test_alias_dispatch_delegates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            import os

            original = os.getcwd()
            os.chdir(root)
            try:
                registry = build_discovery_registry(root)
                result = registry.dispatch("list_files", "list", path=".")
                self.assertEqual(result["status"], "success")
                self.assertTrue(any("src" in i["name"] for i in result["items"]))
                read = registry.dispatch(
                    "read", "read", file_path="src/auth/service.py"
                )
                self.assertIn("AuthenticationService", read["current_page_content"])
                inspected = registry.dispatch(
                    "inspect", "inspect", path="src/auth/service.py"
                )
                self.assertEqual(inspected["language"], "python")
                defined = registry.dispatch(
                    "find_definition",
                    "find_definition",
                    name="authenticate",
                    path="src",
                )
                self.assertEqual(defined["count"], 1)
            finally:
                os.chdir(original)

    def test_registry_dispatch_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            registry = build_discovery_registry(root)
            result = registry.dispatch("list_dir", "list", path=".")
            self.assertEqual(result["status"], "success")
            self.assertTrue(any("src" in i["name"] for i in result["items"]))

    def test_code_spec_unknown_action(self):
        spec = build_code_spec()
        result = spec.dispatch("nope")
        self.assertEqual(result["error"], "unknown_action")


class TestCodeToolset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        _make_project(self.root)

    def test_find_definition(self):
        result = symbols.find_definition(self.root, "authenticate", path="src")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)
        self.assertIn("service.py", result["definitions"][0]["file"])
        self.assertEqual(result["definitions"][0]["kind"], "function")

    def test_find_definition_relative_path(self):
        result = symbols.find_definition(self.root, "AuthenticationService")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)
        definition = result["definitions"][0]
        self.assertTrue(definition["file"].endswith("src/auth/service.py"))
        self.assertEqual(definition["kind"], "class")

    def test_find_references(self):
        result = symbols.find_references(self.root, "authenticate")
        self.assertEqual(result["status"], "success")
        files = [f["file"] for f in result["files"]]
        self.assertTrue(any("api/auth.py" in f for f in files))
        self.assertTrue(any("tests/test_auth.py" in f for f in files))

    def test_find_symbol(self):
        result = symbols.find_symbol(self.root, "authenticate")
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["definitions"])
        self.assertTrue(result["usage_files"])

    def test_find_imports(self):
        result = symbols.find_imports(self.root, "src/api/auth.py")
        self.assertEqual(result["status"], "success")
        self.assertTrue(
            any("src.auth.service" == i["module"] for i in result["imports"])
        )
        self.assertTrue(result["imports"][0]["local"])

    def test_find_importers(self):
        result = symbols.find_importers(self.root, "src/auth/service.py")
        self.assertEqual(result["status"], "success")
        files = [f["file"] for f in result["files"]]
        self.assertTrue(any("api/auth.py" in f for f in files))
        self.assertTrue(any("tests/test_auth.py" in f for f in files))

    def test_inspect(self):
        result = symbols.inspect_file(self.root, "src/auth/service.py")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["language"], "python")
        self.assertIn("Authentication service", result["purpose"])
        self.assertIn("AuthenticationService", [s["symbol"] for s in result["symbols"]])

    def test_missing_file(self):
        result = symbols.find_definition(self.root, "x", path="nope.py")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 0)

    def test_code_spec_dispatch(self):
        spec = build_code_spec(self.root)
        result = spec.dispatch("find_definition", name="authenticate", path="src")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)


class TestDiscoveryAgent(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        _make_project(self.root)

    def test_run_end_to_end(self):
        provider = FakeProvider(
            [
                '{"tool": "list_dir", "action": "list", "params": {"path": "."}}',
                FINAL_REPORT,
            ]
        )
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("find the auth flow")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task"], "find the auth flow")
        self.assertEqual(result["workspace"], str(self.root.resolve()))
        self.assertTrue(result["report"].startswith("DISCOVERY RESULT"))
        metrics = result["metrics"]
        for key in (
            "tool_calls",
            "iterations",
            "files_inspected",
            "files_read",
            "lines_read",
            "candidates_found",
            "candidates_discarded",
            "repeated_calls",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "final_context_tokens",
            "efficiency",
            "elapsed_time",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["iterations"], 2)

    def test_stops_when_report_given(self):
        provider = FakeProvider([FINAL_REPORT])
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["tool_calls"], 0)
        self.assertEqual(result["metrics"]["iterations"], 1)

    def test_uninspected_report_is_fed_back_to_investigate(self):
        thin = (
            "DISCOVERY RESULT\n\nSUMMARY\nDESIGN.md exists but its content "
            "was not yet inspected."
        )
        provider = FakeProvider(
            [thin, '{"tool": "list_dir", "action": "list", "params": {}}', FINAL_REPORT]
        )
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["tool_calls"], 1)
        self.assertEqual(result["metrics"]["iterations"], 3)
        self.assertTrue(result["report"].startswith("DISCOVERY RESULT"))
        self.assertIn("not yet inspected", provider.calls[1][0])

    def test_budget_partial_on_iterations(self):
        provider = FakeProvider(
            ['{"tool": "list_dir", "action": "list", "params": {"path": "."}}'] * 20
        )
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate", max_iterations=3)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["metrics"]["iterations"], 3)
        self.assertIn("iteration limit", result["reason"])
        self.assertTrue(result["report"].startswith("DISCOVERY RESULT"))

    def test_budget_partial_on_tool_calls(self):
        calls = [
            f'{{"tool": "list_dir", "action": "list", "params": {{"path": "d{i}"}}}}'
            for i in range(20)
        ]
        provider = FakeProvider(calls)
        agent = DiscoveryAgent(provider, root=str(self.root))
        from src.discovery.budget import DiscoveryBudget

        agent.budget = DiscoveryBudget(
            max_iterations=100,
            max_tool_calls=2,
            max_context_tokens=40000,
            max_files_read=100,
            max_lines_read=100000,
        )
        result = agent.run("investigate")
        self.assertEqual(result["status"], "partial")
        self.assertIn("tool call limit", result["reason"])
        self.assertEqual(result["metrics"]["tool_calls"], 2)

    def test_requires_request(self):
        provider = FakeProvider()
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_read_budget_counts_lines(self):
        calls = [
            f'{{"tool": "read_file", "action": "read", "params": '
            f'{{"file_path": "src/auth/service.py", "start_line": {i * 5 + 1}, '
            f'"end_line": {i * 5 + 5}}}}}'
            for i in range(20)
        ]
        provider = FakeProvider(calls)
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate", max_lines_read=20)
        self.assertEqual(result["status"], "partial")
        self.assertIn("lines read limit", result["reason"])
        self.assertGreater(result["metrics"]["lines_read"], 0)
        self.assertGreaterEqual(result["metrics"]["files_read"], 1)

    def test_restores_cwd(self):
        import os

        original = os.getcwd()
        provider = FakeProvider([FINAL_REPORT])
        agent = DiscoveryAgent(provider, root=str(self.root))
        agent.run("investigate")
        self.assertEqual(os.getcwd(), original)

    def test_repeated_read_is_skipped_not_recounted(self):
        read_call = (
            '{"tool": "read_file", "action": "read", "params": '
            '{"file_path": "src/auth/service.py"}}'
        )
        provider = FakeProvider([read_call, read_call, FINAL_REPORT])
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["tool_calls"], 1)
        self.assertEqual(result["metrics"]["files_read"], 1)
        self.assertEqual(result["metrics"]["repeated_calls"], 1)
        self.assertIn("Already inspected", provider.calls[1][0])
        self.assertIn("Do NOT re-read", provider.calls[2][0])

    def test_repeated_identical_search_is_skipped(self):
        search_call = '{"tool": "search_files", "action": "search", "params": {"pattern": "*.py"}}'
        provider = FakeProvider([search_call, search_call, FINAL_REPORT])
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["tool_calls"], 1)
        self.assertEqual(result["metrics"]["repeated_calls"], 1)

    def test_repeat_streak_stops_run(self):
        read_call = (
            '{"tool": "read_file", "action": "read", "params": '
            '{"file_path": "src/auth/service.py"}}'
        )
        provider = FakeProvider([read_call, read_call, read_call, read_call])
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate", max_iterations=10)
        self.assertEqual(result["status"], "partial")
        self.assertIn("repeated", result["reason"])
        self.assertEqual(result["metrics"]["repeated_calls"], 3)

    def test_candidates_telemetry(self):
        provider = FakeProvider(
            [
                '{"tool": "list_dir", "action": "list", "params": {"path": "."}}',
                '{"tool": "read_file", "action": "read", "params": {"file_path": "src/auth/service.py"}}',
                FINAL_REPORT,
            ]
        )
        agent = DiscoveryAgent(provider, root=str(self.root))
        result = agent.run("investigate")
        metrics = result["metrics"]
        self.assertGreaterEqual(metrics["candidates_found"], metrics["files_read"])
        self.assertEqual(
            metrics["candidates_discarded"],
            metrics["candidates_found"] - metrics["files_read"],
        )
        self.assertGreaterEqual(metrics["efficiency"], 0)


class TestDiscoveryTool(unittest.TestCase):
    def test_manual_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = create_discovery_tool(FakeProvider([FINAL_REPORT]), root=tmp)
            manual = tool.get_manual()
            self.assertIn("request", manual)
            self.assertLess(manual.index("USE run"), manual.index("list_files"))
            result = tool.dispatch("run", request="investigate")
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["report"].startswith("DISCOVERY RESULT"))
            self.assertIn("metrics", result)

    def test_run_has_no_lookup_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = create_discovery_tool(FakeProvider([FINAL_REPORT]), root=tmp)
            result = tool.dispatch("run", request="investigate")
            self.assertNotIn("hint", result)

    def test_lookup_actions_carry_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([]), root=root)
            for action, params in (
                ("list_files", {"path": "."}),
                ("search", {"pattern": "*.py"}),
                ("grep", {"pattern": "authenticate"}),
                ("read", {"file_path": "src/auth/service.py"}),
                ("inspect", {"path": "src/auth/service.py"}),
            ):
                result = tool.dispatch(action, **params)
                self.assertEqual(result["status"], "success")
                self.assertIn("hint", result)
                self.assertIn("discovery run", result["hint"])

    def test_run_requires_request(self):
        tool = create_discovery_tool(FakeProvider([]), root=".")
        result = tool.dispatch("run")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_run_query_synonym_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([FINAL_REPORT]), root=root)
            result = tool.dispatch("run", query="investigate", path="src/auth")
            self.assertEqual(result["status"], "success")
            self.assertIn("focus: src/auth", result["task"])

    def test_run_question_synonym(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([FINAL_REPORT]), root=root)
            result = tool.dispatch(
                "run",
                question="o que é preciso para aplicar o DESIGN.md no front?",
            )
            self.assertEqual(result["status"], "success")
            self.assertIn(
                "o que é preciso para aplicar o DESIGN.md no front", result["task"]
            )

    def test_list_files_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([]), root=root)
            result = tool.dispatch("list_files", path=".")
            self.assertEqual(result["status"], "success")
            self.assertTrue(any("src" in i["name"] for i in result["items"]))

    def test_read_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([]), root=root)
            result = tool.dispatch("read", file_path="src/auth/service.py")
            self.assertEqual(result["status"], "success")
            self.assertIn("AuthenticationService", result["current_page_content"])

    def test_grep_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([]), root=root)
            result = tool.dispatch("grep", pattern="authenticate")
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["total_matches"], 0)

    def test_inspect_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_project(root)
            tool = create_discovery_tool(FakeProvider([]), root=root)
            result = tool.dispatch("inspect", path="src/auth/service.py")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["language"], "python")

    def test_unknown_action(self):
        tool = create_discovery_tool(FakeProvider([]), root=".")
        result = tool.dispatch("nope")
        self.assertEqual(result["error"], "unknown_action")


if __name__ == "__main__":
    unittest.main()
