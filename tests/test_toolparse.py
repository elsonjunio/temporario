import json
import unittest
from pathlib import Path

from src.toolparse import is_tool_attempt, parse_tool_call

FIXTURES = Path(__file__).parent / "fixtures" / "tool_calls"


class TestToolCallMarkers(unittest.TestCase):
    def test_tool_call_tag_with_call_prefix(self):
        text = '<|tool_call>call:read_file\n{"file_path": "app.py"}\n'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["action"], "read")
        self.assertEqual(call["params"]["file_path"], "app.py")

    def test_tool_call_tag_without_call_prefix(self):
        text = '<|tool_call>list_dir\n{"path": "."}\n'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["params"]["path"], ".")

    def test_bare_json_with_marker(self):
        text = 'Preciso disso. call:run_command\n{"command": "ls -la"}'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "run_command")
        self.assertEqual(call["action"], "run")
        self.assertEqual(call["params"]["command"], "ls -la")

    def test_tool_alias_in_marker(self):
        text = 'call:readfile\n{"file_path": "x.txt"}'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "read_file")

    def test_tool_name_key(self):
        text = 'tool_name: "grep_files"\n{"pattern": "x", "path": "."}'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "grep_files")
        self.assertEqual(call["action"], "search")


class TestSyntaxRepairs(unittest.TestCase):
    def test_single_quotes(self):
        text = "{'tool': 'read_file', 'params': {'file_path': 'a.txt'}}"
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["params"]["file_path"], "a.txt")

    def test_unquoted_keys(self):
        text = "{tool: read_file, file_path: a.txt}"
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["params"]["file_path"], "a.txt")

    def test_trailing_comma(self):
        text = '{"tool": "write_file", "params": {"content": "oi",}}'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "write_file")
        self.assertEqual(call["params"]["content"], "oi")

    def test_truncated_json(self):
        text = '{"tool": "grep_files", "params": {"pattern": "banana", "path": "dados"}'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "grep_files")
        self.assertEqual(call["action"], "search")
        self.assertEqual(call["params"]["pattern"], "banana")

    def test_python_bool_literals(self):
        text = '{"tool": "move_file", "params": {"source": "a", "destination": "b", "create_dirs": True}}'
        call = parse_tool_call(text)
        self.assertIs(call["params"]["create_dirs"], True)

    def test_string_value_keeps_case(self):
        text = '{"tool": "read_file", "params": {"file_path": "True.txt"}}'
        call = parse_tool_call(text)
        self.assertEqual(call["params"]["file_path"], "True.txt")


class TestFuzzyFallback(unittest.TestCase):
    def test_key_value_pairs_in_text(self):
        text = "Preciso buscar. call:grep_files pattern=banana path=dados"
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "grep_files")
        self.assertEqual(call["action"], "search")
        self.assertEqual(call["params"]["pattern"], "banana")

    def test_marker_keyword_not_a_param(self):
        text = "call:grep_files pattern=banana"
        call = parse_tool_call(text)
        self.assertNotIn("call", call["params"])


class TestNonToolAnswers(unittest.TestCase):
    def test_plain_answer_returns_none(self):
        self.assertIsNone(parse_tool_call("Esta e a resposta final."))

    def test_unknown_tool_key_without_marker_returns_none(self):
        self.assertIsNone(parse_tool_call('```json\n{"foo": "bar"}\n```'))

    def test_prose_mentioning_tool_is_not_a_call(self):
        self.assertIsNone(
            parse_tool_call("Usei a ferramenta read_file para ler o arquivo.")
        )

    def test_invoke_without_name_is_not_a_call(self):
        self.assertIsNone(
            parse_tool_call("<invoke>like a normal mention, not a tool call</invoke>")
        )

    def test_unknown_tool_in_json_still_parses(self):
        call = parse_tool_call(
            '```json\n{"tool": "ghost", "action": "x", "params": {}}\n```'
        )
        self.assertEqual(call["tool"], "ghost")


class TestRealMalformedSamples(unittest.TestCase):
    """Fixtures captured live from a small model (gemma-4-e2b) in the sandbox.

    The model emits ``<|tool_call>call:<tool>`` followed by a bare JSON object
    without the ``tool`` key. These must parse without a re-prompt.
    """

    def test_all_fixtures_parse(self):
        with open(FIXTURES / "malformed_samples.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                call = parse_tool_call(sample["input"])
                self.assertIsNotNone(call, f"falhou: {sample['input']!r}")
                self.assertEqual(call["tool"], sample["tool"])
                self.assertEqual(call["action"], sample["action"])


class TestIsToolAttempt(unittest.TestCase):
    def test_tool_call_markers_count_as_attempt(self):
        self.assertTrue(is_tool_attempt("<|tool_call>call:read_file\n..."))
        self.assertTrue(is_tool_attempt("tool: read_file"))
        self.assertTrue(is_tool_attempt("Preciso de call:list_dir"))

    def test_tool_signature_in_json_counts_as_attempt(self):
        self.assertTrue(is_tool_attempt('{"tool": "read_file"}'))
        self.assertTrue(is_tool_attempt("{tool: read_file, file_path: app.py}"))

    def test_bare_braces_alone_are_not_an_attempt(self):
        self.assertFalse(is_tool_attempt("algo { malformado"))

    def test_json_without_tool_signature_is_not_attempt(self):
        self.assertFalse(is_tool_attempt('{"nome": "sandbox", "versao": 1.0}'))

    def test_answer_echoing_json_is_not_attempt(self):
        self.assertFalse(
            is_tool_attempt('O arquivo contem:\n```json\n{"nome": "sandbox"}\n```')
        )

    def test_plain_answer_is_not_attempt(self):
        self.assertFalse(is_tool_attempt("O arquivo foi criado com sucesso."))


if __name__ == "__main__":
    unittest.main()
