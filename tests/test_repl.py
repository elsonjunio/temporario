import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.repl import read_turn, setup_readline


class FakeStdin:
    def __init__(self, isatty):
        self._isatty = isatty

    def isatty(self):
        return self._isatty


class SequenceReader:
    """Serves canned lines; returns EOF (empty) when exhausted."""

    def __init__(self, *lines):
        self.lines = list(lines)
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if not self.lines:
            return ""
        return self.lines.pop(0)


class TestReadTurn(unittest.TestCase):
    def test_non_tty_single_line(self):
        reader = SequenceReader("liste o diretorio")
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=False)):
            self.assertEqual(read_turn(reader=reader), "liste o diretorio")
        self.assertEqual(reader.calls, ["You: "])

    def test_non_tty_eof_returns_none(self):
        reader = SequenceReader()
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=False)):
            self.assertIsNone(read_turn(reader=reader))

    def test_tty_multiline_joined(self):
        reader = SequenceReader("crie um script", "que leia app.py", "")
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=True)):
            self.assertEqual(
                read_turn(reader=reader), "crie um script\nque leia app.py"
            )
        self.assertEqual(reader.calls, ["You: ", "... ", "... "])

    def test_tty_empty_first_line_returns_none(self):
        reader = SequenceReader("")
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=True)):
            self.assertIsNone(read_turn(reader=reader))

    def test_tty_eof_returns_none(self):
        reader = SequenceReader()
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=True)):
            self.assertIsNone(read_turn(reader=reader))

    def test_tty_eof_after_text_returns_collected(self):
        reader = SequenceReader("primeira linha")
        with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=True)):
            self.assertEqual(read_turn(reader=reader), "primeira linha")

    def test_custom_reader_never_writes_history(self):
        reader = SequenceReader("prompt", "")
        with tempfile.TemporaryDirectory() as tmp:
            history = str(Path(tmp) / "hist")
            with mock.patch("src.repl.sys.stdin", FakeStdin(isatty=True)):
                read_turn(reader=reader, history=history)
            self.assertFalse(Path(history).exists())


class TestSetupReadline(unittest.TestCase):
    def test_missing_history_file_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_readline(str(Path(tmp) / "nao-existe"), register_atexit=False)
