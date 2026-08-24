"""Tests for the patcher subagent: sandboxed path rules, deterministic
validation gates (in-memory dry run + no-op rejection), multi-file
generation without ever writing, and the standalone ``patcher`` tool."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.patcher import (
    PatcherAgent,
    PatchGenerator,
    create_patcher_tool,
    sandbox_path,
)


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def infer(self, user_prompt, config=None, **settings):
        self.calls.append((user_prompt, config))
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("unexpected extra provider call")


def _replace(old: str, new: str) -> str:
    return json.dumps({"mode": "replace", "old": old, "new": new})


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "f.txt").write_text("hello world\n")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "x.js").write_text("x\n")
        (self.root / "bin.dat").write_bytes(b"\x00\x01binary")

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_path_resolves_inside_root(self):
        resolved, error = sandbox_path(str(self.root), "f.txt")
        self.assertIsNone(error)
        assert resolved is not None
        self.assertTrue(Path(resolved).is_absolute())

    def test_traversal_outside_root_rejected(self):
        resolved, error = sandbox_path(str(self.root), "../outside.txt")
        self.assertIsNone(resolved)
        self.assertIn("outside the workspace root", error)

    def test_absolute_outside_root_rejected(self):
        resolved, error = sandbox_path(str(self.root), "/etc/hosts")
        self.assertIsNone(resolved)
        self.assertIn("outside the workspace root", error)

    def test_excluded_directory_rejected(self):
        resolved, error = sandbox_path(str(self.root), "node_modules/x.js")
        self.assertIsNone(resolved)
        self.assertIn("excluded directory", error)

    def test_missing_file_rejected(self):
        resolved, error = sandbox_path(str(self.root), "nope.txt")
        self.assertIsNone(resolved)
        self.assertIn("file not found", error)

    def test_binary_file_rejected(self):
        resolved, error = sandbox_path(str(self.root), "bin.dat")
        self.assertIsNone(resolved)
        self.assertIn("binary file", error)

    def test_oversize_file_rejected(self):
        big = self.root / "big.txt"
        big.write_text("x" * 64)
        with mock.patch.dict(os.environ, {"PATCHER_MAX_FILE_BYTES": "16"}):
            resolved, error = sandbox_path(str(self.root), "big.txt")
        self.assertIsNone(resolved)
        self.assertIn("too large", error)


class TestPatchGeneratorGates(unittest.TestCase):
    def setUp(self):
        self.provider_calls: list[tuple[str, str]] = []

        class Provider:
            def infer(_self, user_prompt, config=None, **settings):
                self.provider_calls.append((user_prompt, config or ""))
                return self.next_response

        self.provider = Provider()

    def test_noop_replace_rejected(self):
        error, params = PatchGenerator._validate(
            {"mode": "replace", "old": "a", "new": "a"}, "a\n"
        )
        self.assertIn("unchanged", error)
        self.assertEqual(params, {})

    def test_noop_diff_rejected(self):
        diff = "@@ -1,1 +1,1 @@\n-hello\n+hello\n"
        error, _ = PatchGenerator._validate({"mode": "apply", "diff": diff}, "hello\n")
        self.assertIn("unchanged", error)

    def test_overlapping_hunks_fail_dry_run(self):
        diff = "@@ -1,2 +1,0 @@\n-a\n-b\n@@ -2,2 +0,0 @@\n-b\n-c\n"
        error, _ = PatchGenerator._validate(
            {"mode": "apply", "diff": diff}, "a\nb\nc\n"
        )
        self.assertIsNotNone(error)
        self.assertTrue(
            error.startswith("hunks do not apply cleanly") or "not found" in error,
            msg=error,
        )

    def test_valid_apply_passes_dry_run(self):
        diff = "@@ -1,1 +1,1 @@\n-hello\n+goodbye\n"
        error, params = PatchGenerator._validate(
            {"mode": "apply", "diff": diff}, "hello\n"
        )
        self.assertIsNone(error)
        self.assertEqual(params["diff"], diff)

    def test_instruction_is_capped_in_prompt(self):
        self.next_response = _replace("hello", "goodbye")
        with mock.patch.dict(os.environ, {"PATCHER_MAX_INSTRUCTION_CHARS": "12"}):
            generated = PatchGenerator(self.provider).generate(
                "f.txt", "hello\n", "x" * 200
            )
        self.assertEqual(generated["status"], "ok")
        prompt, _config = self.provider_calls[0]
        self.assertIn("x" * 12, prompt)
        self.assertNotIn("x" * 13, prompt)

    def test_prompt_isolates_context_and_marks_content_as_data(self):
        self.next_response = _replace("hello", "goodbye")
        generated = PatchGenerator(self.provider).generate(
            "f.txt", "hello\n", "swap greeting"
        )
        self.assertEqual(generated["status"], "ok")
        prompt, config = self.provider_calls[0]
        self.assertNotIn("Available tools", prompt)
        self.assertNotIn('"steps"', prompt)
        self.assertIn("swap greeting", prompt)
        self.assertIn("1 | hello", prompt)
        lowered = config.lower()
        self.assertIn("verbatim", lowered)
        self.assertIn("data", lowered)


class TestPatcherAgentPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.a = self.root / "a.txt"
        self.b = self.root / "b.txt"
        self.a.write_text("hello there\n")
        self.b.write_text("second hello\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, responses) -> tuple[PatcherAgent, FakeProvider]:
        provider = FakeProvider(responses)
        return PatcherAgent(provider, root=str(self.root)), provider

    def test_multi_file_generation_never_writes(self):
        agent, provider = self._agent(
            [_replace("hello", "goodbye"), _replace("hello", "hi")]
        )
        result = agent.run('replace "hello"', paths=["a.txt", "b.txt"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["patches"]), 2)
        self.assertEqual(result["patches"][0]["file_path"], str(self.a))
        self.assertEqual(result["patches"][0]["params"]["file_path"], str(self.a))
        # Generation is separate from application: disk untouched.
        self.assertIn("hello", self.a.read_text())
        self.assertIn("hello", self.b.read_text())
        # Each generation saw ONLY its own file content.
        prompt_a, _ = provider.calls[0]
        prompt_b, _ = provider.calls[1]
        self.assertIn("hello there", prompt_a)
        self.assertNotIn("second hello", prompt_a)
        self.assertIn("second hello", prompt_b)
        self.assertNotIn("hello there", prompt_b)

    def test_partial_when_one_file_fails(self):
        agent, _provider = self._agent(
            [
                _replace("hello", "goodbye"),
                _replace("nope-anchor", "x"),
                _replace("nope-anchor-2", "y"),
            ]
        )
        result = agent.run("edit greetings", paths=["a.txt", "b.txt"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["patches"]), 1)
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("not found", result["failures"][0]["message"])

    def test_invalid_path_becomes_failure_not_crash(self):
        agent, _provider = self._agent([_replace("hello", "goodbye")])
        result = agent.run("edit", paths=["missing.txt", "a.txt"])
        self.assertEqual(result["status"], "partial")
        self.assertIn("file not found", result["failures"][0]["message"])

    def test_max_files_patched_limit(self):
        agent, _provider = self._agent([_replace("hello", "goodbye")])
        result = agent.run("edit", paths=["a.txt", "b.txt"], max_files_patched=1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["patches"]), 1)
        self.assertIn("max_files_patched", result["failures"][0]["message"])

    def test_weak_model_non_json_exhausts_retries(self):
        agent, provider = self._agent(["not json", "still not json"])
        result = agent.run("edit a.txt", paths=["a.txt"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("invalid patch", result["failures"][0]["message"])

    def test_context_is_passed_capped(self):
        agent, provider = self._agent([_replace("hello", "goodbye")])
        result = agent.run("edit", paths=["a.txt"], context="keep the tone casual")
        self.assertEqual(result["status"], "ok")
        prompt, _ = provider.calls[0]
        self.assertIn("Additional context from the caller:", prompt)
        self.assertIn("keep the tone casual", prompt)

    def test_empty_instruction_is_invalid_arguments(self):
        agent, _provider = self._agent([])
        result = agent.run("   ")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")


class TestPatcherAgentLocate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        src = self.root / "src"
        src.mkdir()
        (src / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.root / "notes.md").write_text("# notes\nnothing here\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_locate_then_generate(self):
        grep_call = json.dumps(
            {
                "tool": "grep_files",
                "action": "search",
                "params": {"pattern": "def add"},
            }
        )
        final_json = json.dumps(
            {"files": [{"path": "src/calc.py", "why": "add implementation is wrong"}]}
        )
        provider = FakeProvider([grep_call, final_json, _replace("a - b", "a + b")])
        agent = PatcherAgent(provider, root=str(self.root))
        result = agent.run("fix the add function: it subtracts instead of adding")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["located"]), 1)
        self.assertEqual(
            result["located"][0]["path"],
            str(self.root / "src" / "calc.py"),
        )
        self.assertEqual(result["located"][0]["why"], "add implementation is wrong")
        self.assertEqual(len(result["patches"]), 1)
        self.assertGreaterEqual(result["metrics"]["tool_calls"], 1)
        # The generation prompt carried the located rationale.
        generate_prompt = provider.calls[2][0]
        self.assertIn("target rationale: add implementation is wrong", generate_prompt)

    def test_hallucinated_located_path_is_filtered(self):
        final_json = json.dumps({"files": ["src/ghost.py", {"path": "src/calc.py"}]})
        provider = FakeProvider([final_json, _replace("a - b", "a + b")])
        agent = PatcherAgent(provider, root=str(self.root))
        result = agent.run("fix add")
        self.assertEqual(result["status"], "partial")
        self.assertIn("file not found", result["failures"][0]["message"])
        self.assertEqual(len(result["patches"]), 1)

    def test_locator_without_final_json_errors(self):
        provider = FakeProvider(["let me think...", "still thinking...", "hmm..."])
        agent = PatcherAgent(provider, root=str(self.root))
        result = agent.run("fix something")
        self.assertEqual(result["status"], "error")
        self.assertIn("locate_reason", result)
        self.assertEqual(result["patches"], [])

    def test_locate_system_prompt_marks_files_as_data(self):
        provider = FakeProvider(
            [
                json.dumps({"files": []}),
            ]
        )
        agent = PatcherAgent(provider, root=str(self.root))
        agent.run("anything")
        _prompt, config = provider.calls[0]
        lowered = config.lower()
        self.assertIn("read-only", lowered)
        self.assertIn("data", lowered)


class TestCreatePatcherTool(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "f.txt").write_text("hello world\n")
        self.tool = create_patcher_tool(FakeProvider(), root=str(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_validate_ok_returns_sanitized_params(self):
        result = self.tool.dispatch(
            "validate", file_path="f.txt", old="hello", new="goodbye"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "replace")
        self.assertEqual(result["params"]["file_path"], str(self.root / "f.txt"))
        self.assertEqual(result["params"]["old"], "hello")

    def test_validate_rejects_bad_anchor(self):
        result = self.tool.dispatch("validate", file_path="f.txt", old="ghost", new="x")
        self.assertEqual(result["status"], "rejected")
        self.assertIn("not found", result["message"])

    def test_validate_requires_payload(self):
        result = self.tool.dispatch("validate", file_path="f.txt")
        self.assertEqual(result["status"], "error")
        self.assertIn("either old/new", result["message"])

    def test_validate_sandbox_error(self):
        result = self.tool.dispatch(
            "validate", file_path="../escape.txt", old="a", new="b"
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("outside the workspace root", result["message"])

    def test_run_without_instruction_is_clean_error(self):
        result = self.tool.dispatch("run", paths=["f.txt"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    def test_run_via_synonym_action_and_request_key(self):
        tool = create_patcher_tool(
            FakeProvider([_replace("hello", "goodbye")]), root=str(self.root)
        )
        result = tool.dispatch("generate", request="change hello", paths="f.txt")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["patches"]), 1)

    def test_unknown_action(self):
        result = self.tool.dispatch("explode")
        self.assertEqual(result["error"], "unknown_action")


if __name__ == "__main__":
    unittest.main()
