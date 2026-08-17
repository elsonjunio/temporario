import unittest

from src.main import parse_args
from src.utils import DEFAULT_MODE


class TestMainCli(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.max_iterations, 5)
        self.assertEqual(args.max_plan_steps, 16)
        self.assertIsNone(args.prompt)
        self.assertEqual(args.mode, DEFAULT_MODE)

    def test_custom_flags(self):
        args = parse_args(
            [
                "--root",
                "/tmp/projeto",
                "--max-iterations",
                "15",
                "--max-plan-steps",
                "12",
                "--prompt",
                "crie o projeto",
            ]
        )
        self.assertEqual(args.root, "/tmp/projeto")
        self.assertEqual(args.max_iterations, 15)
        self.assertEqual(args.max_plan_steps, 12)
        self.assertEqual(args.prompt, "crie o projeto")

    def test_mode_flag(self):
        args = parse_args(["--mode", "precision"])
        self.assertEqual(args.mode, "precision")

    def test_mode_flag_rejects_unknown(self):
        with self.assertRaises(SystemExit):
            parse_args(["--mode", "turbo"])
