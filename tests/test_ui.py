import io
import unittest
from unittest import mock

from rich.console import Console

from src.ui import ChatUI


class FakeSession:
    """Stands in for a prompt_toolkit PromptSession in tests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def prompt(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if not self.responses:
            raise EOFError
        return self.responses.pop(0)


class TestChatUI(unittest.TestCase):
    def _ui(self, session=None, **kwargs):
        console = Console(record=True, file=io.StringIO())
        return ChatUI(console=console, session=session, **kwargs)

    def test_read_turn_returns_text(self):
        ui = self._ui(session=FakeSession("oi"))
        self.assertEqual(ui.read_turn(), "oi")

    def test_read_turn_multiline_and_toolbar(self):
        session = FakeSession("oi")
        ui = self._ui(session=session, model="big-pickle", workspace="/tmp/ws")
        ui.read_turn()
        prompt, kwargs = session.calls[0]
        self.assertEqual(prompt, "Você: ")
        self.assertTrue(kwargs["multiline"])
        self.assertIsNotNone(kwargs["bottom_toolbar"])

    def test_read_turn_eof_returns_none(self):
        ui = self._ui(session=FakeSession())
        self.assertIsNone(ui.read_turn())

    def test_read_turn_non_interactive_uses_plain_input(self):
        ui = self._ui(interactive=False)
        self.assertIsNone(ui.session)
        with mock.patch("builtins.input", return_value="oi"):
            self.assertEqual(ui.read_turn(), "oi")

    def test_read_turn_non_interactive_eof(self):
        ui = self._ui(interactive=False)
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(ui.read_turn())

    def test_banner_shows_model_and_workspace(self):
        ui = self._ui(model="big-pickle", workspace="/tmp/ws")
        ui.banner()
        text = ui.console.export_text()
        self.assertIn("Bicicleta com Rodinhas", text)
        self.assertIn("big-pickle", text)
        self.assertIn("/tmp/ws", text)

    def test_show_assistant_panels_answer(self):
        ui = self._ui()
        ui.show_assistant("Olá!")
        self.assertIn("Olá!", ui.console.export_text())

    def test_show_error(self):
        ui = self._ui()
        ui.show_error("boom")
        self.assertIn("boom", ui.console.export_text())

    def test_show_help_lists_commands(self):
        ui = self._ui()
        ui.show_help()
        text = ui.console.export_text()
        self.assertIn("/quit", text)
        self.assertIn("/clear", text)

    def test_working_status_yields(self):
        ui = self._ui()
        with ui.working("Pensando..."):
            pass


if __name__ == "__main__":
    unittest.main()
