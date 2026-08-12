import unittest

from src.agent import Agent
from src.memory import Memory
from src.tools.registry import build_default_registry


class FakeProvider:
    """Serves canned responses to drive the agent loop without a network."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


class TestAgentLoop(unittest.TestCase):
    def test_plain_answer(self):
        provider = FakeProvider(["Final answer."])
        agent = Agent(provider=provider)
        result = agent.run("hi")
        self.assertEqual(result, "Final answer.")
        self.assertEqual(len(provider.calls), 1)

    def test_tool_call_then_answer(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "list_dir", "action": "list", "params": {"path": "src"}}\n```',
                "Here is the listing.",
            ]
        )
        memory = Memory()
        agent = Agent(provider=provider, memory=memory)
        result = agent.run("list src")
        self.assertEqual(result, "Here is the listing.")
        self.assertEqual(len(provider.calls), 2)
        # tool result recorded in memory
        history = memory.get_history()
        roles = [entry["role"] for entry in history]
        self.assertIn("tool", roles)

    def test_max_iterations(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "list_dir", "action": "list", "params": {}}\n```',
                '```json\n{"tool": "list_dir", "action": "list", "params": {}}\n```',
            ]
        )
        agent = Agent(provider=provider, max_iterations=2)
        result = agent.run("loop")
        self.assertIn("maximum of 2 tool iterations", result)

    def test_unknown_tool_dispatched(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "ghost", "action": "x", "params": {}}\n```',
                "done",
            ]
        )
        agent = Agent(provider=provider)
        result = agent.run("go")
        self.assertEqual(result, "done")

    def test_extra_tool_registered_is_available(self):
        dummy = type(
            "Dummy",
            (),
            {
                "name": "dummy",
                "get_manual": lambda self: "dummy manual\nActions:\nx",
                "dispatch": lambda self, action, **p: {"status": "success"},
            },
        )()
        provider = FakeProvider(
            [
                '```json\n{"tool": "dummy", "action": "ping", "params": {}}\n```',
                "pong",
            ]
        )
        registry = build_default_registry()
        agent = Agent(provider=provider, registry=registry, extra_tools=[dummy])
        result = agent.run("ping the dummy")
        self.assertEqual(result, "pong")
        self.assertIn("dummy", agent.registry.list_tools())

    def test_agent_works_without_orchestrator(self):
        registry = build_default_registry()
        provider = FakeProvider(["plain answer"])
        agent = Agent(provider=provider, registry=registry)
        self.assertEqual(agent.run("hi"), "plain answer")
        self.assertNotIn("orchestrator", agent.registry.list_tools())


if __name__ == "__main__":
    unittest.main()
