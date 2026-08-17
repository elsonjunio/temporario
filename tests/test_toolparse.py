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

    def test_colon_fused_tool_and_action(self):
        call = parse_tool_call(
            '{"tool": "discovery:list_files", "action": "list_files", "params": {"path": "."}}'
        )
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "list_files")
        self.assertEqual(call["params"]["path"], ".")

    def test_colon_fused_tool_action_without_action_key(self):
        call = parse_tool_call(
            '{"tool": "discovery:list_files", "params": {"path": "."}}'
        )
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "list_files")

    def test_colon_fused_read_tool(self):
        call = parse_tool_call(
            '{"tool": "read_file:read", "params": {"file_path": "a.py"}}'
        )
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["action"], "read")

    def test_marker_with_tool_action_suffix(self):
        call = parse_tool_call(
            '<|tool_call>call:discovery:read\n{"file_path": "/abs/DESIGN.md"}'
        )
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "read")
        self.assertEqual(call["params"]["file_path"], "/abs/DESIGN.md")

    def test_marker_with_tool_action_suffix_fuzzy(self):
        call = parse_tool_call('<|tool_call>call:discovery:grep pattern="--color"')
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "grep")
        self.assertEqual(call["params"]["pattern"], "--color")

    def test_bare_fused_marker_with_json(self):
        call = parse_tool_call('list_dir:list\n{"path": "/x", "recursive": true}')
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["action"], "list")
        self.assertEqual(call["params"]["path"], "/x")
        self.assertIs(call["params"]["recursive"], True)

    def test_bare_fused_marker_followed_by_param_lines(self):
        call = parse_tool_call(
            "discovery:run\n"
            "    request: o que é preciso para aplicar o DESIGN.md no front e "
            "quais arquivos devem receber ajustes"
        )
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "run")
        self.assertEqual(
            call["params"]["request"],
            "o que é preciso para aplicar o DESIGN.md no front e quais "
            "arquivos devem receber ajustes",
        )
        self.assertNotIn("discovery", call["params"])

    def test_tool_name_as_single_top_level_key(self):
        call = parse_tool_call(
            '{"write_file": {"file_path": "frontend/src/styles.css", '
            '"content": "x", "rewrite": true}}'
        )
        self.assertEqual(call["tool"], "write_file")
        self.assertEqual(call["action"], "write")
        self.assertEqual(call["params"]["file_path"], "frontend/src/styles.css")
        self.assertIs(call["params"]["rewrite"], True)

    def test_single_key_non_tool_object_is_not_a_call(self):
        self.assertIsNone(parse_tool_call('{"path": "/x", "recursive": true}'))

    def test_marker_multiline_params_keep_full_values(self):
        text = (
            "<|tool_call>call:discovery:run\n"
            "    request: Investigue o projeto X e produza relatório, "
            "incluindo pendências.\n"
            '    paths: ["/abs/proj", "/abs/docs"]\n'
            "    confirm: true"
        )
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "discovery")
        self.assertEqual(call["action"], "run")
        self.assertEqual(
            call["params"]["request"],
            "Investigue o projeto X e produza relatório, incluindo pendências.",
        )
        self.assertEqual(call["params"]["paths"], ["/abs/proj", "/abs/docs"])
        self.assertIs(call["params"]["confirm"], True)

    def test_marker_multiline_does_not_hijack_prose_keyval(self):
        text = (
            "<|tool_call>call:discovery:run\n"
            "    request: Investigue o projeto. Não altere arquivos: use o "
            "discovery.\n"
            "    confirm: false"
        )
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "discovery")
        self.assertIn(
            "Não altere arquivos: use o discovery.", call["params"]["request"]
        )
        self.assertNotIn("arquivos", call["params"])
        self.assertIs(call["params"]["confirm"], False)

    def test_invoke_with_string_attribute(self):
        call = parse_tool_call(
            '<tool_calls>\n<invoke name="list_dir">\n'
            '<parameter name="path" string="true">.</parameter>\n'
            '<parameter name="tree" string="false">true</parameter>\n'
            '<parameter name="max_depth" string="false">3</parameter>\n'
            "</invoke>\n</tool_calls>"
        )
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["params"]["path"], ".")
        self.assertIs(call["params"]["tree"], True)
        self.assertEqual(call["params"]["max_depth"], 3)

    def test_fullwidth_artifact_invoke(self):
        pipe = "\uff5c"
        call = parse_tool_call(
            f"<{pipe}{pipe}DSM{pipe}{pipe}tool_calls>\n"
            f'<{pipe}{pipe}DSM{pipe}{pipe}invoke name="read_file">\n'
            f'<{pipe}{pipe}DSM{pipe}{pipe}parameter name="file_path" string="true">'
            f"DESIGN.md</{pipe}{pipe}DSM{pipe}{pipe}parameter>\n"
            f"</{pipe}{pipe}DSM{pipe}{pipe}invoke>\n"
            f"</{pipe}{pipe}DSM{pipe}{pipe}tool_calls>"
        )
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["params"]["file_path"], "DESIGN.md")

    def test_prose_artifact_brace_is_not_mangled(self):
        self.assertIsNone(parse_tool_call("<|a|b> some prose"))


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
