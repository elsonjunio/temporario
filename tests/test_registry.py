import unittest

from src.tools import registry as tool_registry
from src.tools.registry import ToolRegistry, build_default_registry


class TestToolRegistry(unittest.TestCase):
    def test_default_registry_has_base_tools(self):
        self.assertEqual(
            set(tool_registry.default_registry.list_tools()),
            {
                "read_file",
                "list_dir",
                "search_files",
                "write_file",
                "patch_file",
                "delete_file",
                "move_file",
                "run_command",
            },
        )

    def test_build_returns_fresh_registry(self):
        reg = build_default_registry()
        self.assertIn("read_file", reg.list_tools())
        self.assertIn("write_file", reg.list_tools())

    def test_register_and_unregister(self):
        reg = ToolRegistry()
        dummy = type(
            "Dummy",
            (),
            {
                "get_manual": lambda self: "dummy manual\nActions:\nx",
                "dispatch": lambda self, action, **p: {"ok": True},
            },
        )()
        reg.register("dummy", dummy)
        self.assertIn("dummy", reg.list_tools())
        self.assertIn("dummy", reg.get_manual())
        self.assertTrue(reg.unregister("dummy"))
        self.assertNotIn("dummy", reg.list_tools())

    def test_unregister_missing_returns_false(self):
        reg = ToolRegistry()
        self.assertFalse(reg.unregister("ghost"))

    def test_dispatch_unknown_tool(self):
        result = tool_registry.default_registry.dispatch("ghost", "x")
        self.assertEqual(result["error"], "unknown_tool")

    def test_unregister_returns_to_base_tools(self):
        reg = build_default_registry()
        reg.register(
            "orchestrator",
            type(
                "O",
                (),
                {
                    "get_manual": lambda self: "orchestrator manual\nActions:\nx",
                    "dispatch": lambda self, action, **p: {"ok": True},
                },
            )(),
        )
        self.assertIn("orchestrator", reg.list_tools())
        self.assertTrue(reg.unregister("orchestrator"))
        self.assertNotIn("orchestrator", reg.list_tools())
        self.assertIn("write_file", reg.list_tools())


if __name__ == "__main__":
    unittest.main()
