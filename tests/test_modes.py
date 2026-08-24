import tempfile
import unittest
from pathlib import Path

from src.modes import (
    MODE_DESCRIPTIONS,
    MODE_NAMES,
    SUBAGENT_NAMES,
    _report_candidates,
    apply_mode,
    make_discovery_fallback,
)
from src.tools.registry import build_default_registry


class FakeSpec:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_manual(self) -> str:
        return f"{self.name} manual"

    def dispatch(self, action: str, **params):
        return {"status": "success", "action": action, "params": params}


def _fake_specs() -> dict:
    specs = {"navigation": FakeSpec("browser")}
    for name in SUBAGENT_NAMES:
        specs[name] = FakeSpec(name)
    return specs


class TestApplyMode(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()
        self.specs = _fake_specs()

    def test_orchestrator_in_fast_and_balanced_only(self):
        for mode in ("fast", "balanced"):
            tools = apply_mode(self.registry, mode, self.specs)
            self.assertIn("orchestrator", tools)
            self.assertIn("browser", tools)
            for base in (
                "read_file",
                "run_command",
                "write_file",
                "search_files",
            ):
                self.assertIn(base, tools)
            for sub in ("discovery", "planner", "executor"):
                self.assertNotIn(sub, tools)

    def test_precision_has_no_subagents(self):
        tools = apply_mode(self.registry, "precision", self.specs)
        self.assertNotIn("orchestrator", tools)
        self.assertIn("browser", tools)
        for base in (
            "read_file",
            "run_command",
            "write_file",
            "search_files",
        ):
            self.assertIn(base, tools)

    def test_switching_modes_is_idempotent(self):
        apply_mode(self.registry, "fast", self.specs)
        apply_mode(self.registry, "fast", self.specs)
        self.assertIn("orchestrator", self.registry.list_tools())
        apply_mode(self.registry, "precision", self.specs)
        apply_mode(self.registry, "precision", self.specs)
        tools = self.registry.list_tools()
        self.assertNotIn("planner", tools)
        self.assertNotIn("orchestrator", tools)
        # back to a subagent mode re-registers it
        apply_mode(self.registry, "balanced", self.specs)
        self.assertIn("orchestrator", self.registry.list_tools())

    def test_base_tools_always_present(self):
        apply_mode(self.registry, "precision", self.specs)
        for tool in ("read_file", "run_command", "browser"):
            self.assertIn(tool, self.registry.list_tools())

    def test_mode_tables_are_coherent(self):
        self.assertEqual(MODE_NAMES, ("fast", "balanced", "precision"))
        self.assertEqual(set(MODE_DESCRIPTIONS), set(MODE_NAMES))


class TestBuildAgentSpecs(unittest.TestCase):
    def test_specs_include_working_orchestrator(self):
        from src.modes import build_agent_specs

        registry = build_default_registry()

        class FakeProvider:
            model = "fake"

            def infer(self, prompt, config=None):  # pragma: no cover
                return "ok"

        specs = build_agent_specs(
            registry,
            FakeProvider(),
            None,
            root=".",
        )
        self.assertIn("navigation", specs)
        self.assertIn("orchestrator", specs)
        orch = specs["orchestrator"]
        self.assertEqual(orch.name, "orchestrator")
        apply_mode(registry, "fast", specs)
        result = registry.dispatch("orchestrator", "pending")
        self.assertEqual(result["status"], "no_pending_plan")


class TestReportCandidates(unittest.TestCase):
    def test_parses_relevant_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.txt"
            notes.write_text("hi\n")
            report = (
                "DISCOVERY RESULT\n\n"
                "TASK\nfind the notes\n\n"
                "SUMMARY\nfound notes.txt\n\n"
                "RELEVANT FILES\n"
                "1. notes.txt\n"
                "   Purpose: main notes file\n"
                "   Relevance: high\n"
                "2. other.txt\n"
                "   Purpose: secondary\n"
                "\n"
                "RELATIONSHIPS\nnone\n"
                "IMPLEMENTATION AREA\nnotes.txt\n"
            )
            candidates = _report_candidates(report, tmp)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0]["name"], "notes.txt")
            self.assertEqual(candidates[0]["type"], "file")
            self.assertIn("main notes file", candidates[0]["snippet"])
            self.assertEqual(candidates[1]["name"], "other.txt")
            self.assertEqual(candidates[1]["type"], "new_file")

    def test_falls_back_to_implementation_area(self):
        report = (
            "RELEVANT FILES\n(none confirmed)\n\n"
            "IMPLEMENTATION AREA\n"
            "config/app.py\n"
            "the files likely to be modified\n"
        )
        candidates = _report_candidates(report, "/tmp")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["path"].endswith("config/app.py"), True)
        self.assertEqual(candidates[0]["type"], "new_file")

    def test_empty_report_yields_no_candidates(self):
        self.assertEqual(_report_candidates("", "/tmp"), [])


class FakeDiscovery:
    name = "discovery"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_manual(self) -> str:
        return "discovery manual"

    def dispatch(self, action: str, **params):
        self.calls.append((action, params))
        return self.result


class TestMakeDiscoveryFallback(unittest.TestCase):
    def test_noop_when_discovery_not_registered(self):
        registry = build_default_registry()
        fallback = make_discovery_fallback(registry)
        result = fallback("find x", None, None)
        self.assertEqual(result["status"], "poor")
        self.assertIn("not registered", result["reason"])

    def test_converts_report_to_evidence(self):
        registry = build_default_registry()
        report = "RELEVANT FILES\n" "1. notes.txt\n" "   Purpose: the target file\n"
        fake = FakeDiscovery(
            {"status": "success", "task": "find", "report": report, "metrics": {}}
        )
        registry.register("discovery", fake)
        fallback = make_discovery_fallback(registry, root="/tmp")
        result = fallback("find the notes", None, None)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "discovery_subagent")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["candidates"][0]["name"], "notes.txt")
        self.assertEqual(fake.calls[0][0], "run")

    def test_subagent_error_returns_poor(self):
        registry = build_default_registry()
        fake = FakeDiscovery(
            {
                "status": "error",
                "error": {"type": "discovery_error", "message": "boom"},
            }
        )
        registry.register("discovery", fake)
        fallback = make_discovery_fallback(registry)
        result = fallback("find x", None, None)
        self.assertEqual(result["status"], "poor")
        self.assertEqual(result["reason"], "boom")


if __name__ == "__main__":
    unittest.main()
