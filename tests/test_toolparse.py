from __future__ import annotations

import unittest

from src.utils import is_tool_attempt, parse_tool_call


class TestParseToolCallTolerant(unittest.TestCase):
    def test_single_quotes(self):
        call = parse_tool_call(
            "{'tool': 'read_file', 'action': 'read', 'params': {'file_path': 'a.py'}}"
        )
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["params"]["file_path"], "a.py")

    def test_unquoted_keys(self):
        call = parse_tool_call(
            '{tool: "list_dir", action: "list", params: {path: "."}}'
        )
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["action"], "list")
        self.assertEqual(call["params"]["path"], ".")

    def test_python_literals(self):
        call = parse_tool_call(
            '{"tool": "write_file", "params": {"overwrite": True, "count": None}}'
        )
        self.assertTrue(call["params"]["overwrite"])
        self.assertIsNone(call["params"]["count"])

    def test_trailing_comma(self):
        call = parse_tool_call('{"tool": "list_dir", "params": {"path": ".",}}')
        self.assertEqual(call["tool"], "list_dir")

    def test_truncated_json(self):
        call = parse_tool_call(
            '{"tool": "grep_files", "action": "search", "params": {"pattern": "import"'
        )
        self.assertEqual(call["tool"], "grep_files")
        self.assertEqual(call["params"]["pattern"], "import")

    def test_tool_from_call_marker(self):
        call = parse_tool_call('call:read_file file_path="a.py"')
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["action"], "read")
        self.assertEqual(call["params"]["file_path"], "a.py")

    def test_tool_from_tool_call_marker(self):
        call = parse_tool_call('<|tool_call|>run_command command="ls -la"')
        self.assertEqual(call["tool"], "run_command")
        self.assertEqual(call["params"]["command"], "ls -la")

    def test_tool_alias_normalized(self):
        call = parse_tool_call('{"tool": "readfile", "params": {"file_path": "x"}}')
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["action"], "read")

    def test_grep_file_alias(self):
        call = parse_tool_call('{"tool": "grep_file", "params": {"pattern": "foo"}}')
        self.assertEqual(call["tool"], "grep_files")
        self.assertEqual(call["action"], "search")

    def test_default_action_when_omitted(self):
        call = parse_tool_call('{"tool": "patch_file", "params": {}}')
        self.assertEqual(call["action"], "apply")

    def test_extra_keys_become_params(self):
        call = parse_tool_call('{"tool": "run_command", "command": "ls", "cwd": "."}')
        self.assertEqual(call["params"]["command"], "ls")
        self.assertEqual(call["params"]["cwd"], ".")

    def test_missing_tool_key_returns_none(self):
        self.assertIsNone(parse_tool_call('{"foo": "bar"}'))

    def test_fuzzy_key_value_fallback(self):
        call = parse_tool_call('tool_name: search_files query="def main"')
        self.assertEqual(call["tool"], "search_files")
        self.assertEqual(call["params"]["query"], "def main")


class TestIsToolAttempt(unittest.TestCase):
    def test_plain_answer_with_braces_is_not_attempt(self):
        self.assertFalse(is_tool_attempt('the file contains {"a": 1} in it'))

    def test_known_tool_in_marker_is_attempt(self):
        self.assertTrue(is_tool_attempt("call:list_dir path=/tmp"))

    def test_json_with_tool_key_is_attempt(self):
        self.assertTrue(is_tool_attempt('{"tool": "run_command", "command": "ls"}'))

    def test_plain_answer_is_not_attempt(self):
        self.assertFalse(is_tool_attempt("tudo pronto, sem chamadas."))


if __name__ == "__main__":
    unittest.main()
