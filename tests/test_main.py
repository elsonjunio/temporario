import unittest

from src.main import parse_args


class TestMainCli(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.max_iterations, 5)
        self.assertEqual(args.max_plan_steps, 8)
        self.assertIsNone(args.prompt)

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
