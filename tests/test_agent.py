import unittest
from unittest import mock

from src.agent import Agent
from src.memory import Memory
from src.tools.registry import build_default_registry
from src.utils import AGENT_MODES


class FakeProvider:
    """Serves canned responses to drive the agent loop without a network."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, user_prompt, config, **settings):
        self.calls.append((user_prompt, config))
        return self.responses.pop(0)


class FakeOrchestrator:
    """Stands in for the real orchestrator to exercise the confirm gate."""

    def __init__(self):
        self.calls = []

    def get_manual(self):
        return "orchestrator manual\nActions:\n  - run\n  - execute\n  - abort"

    def dispatch(self, action, **params):
        self.calls.append((action, params))
        if action == "run":
            return {
                "status": "awaiting_confirmation",
                "message": "approval needed",
                "summary": [
                    {
                        "step": 1,
                        "tool": "write_file",
                        "action": "write",
                        "description": "write out.txt",
                        "params": {"file_path": "out.txt"},
                    }
                ],
                "steps": [
                    {
                        "tool": "write_file",
                        "action": "write",
                        "params": {"file_path": "out.txt"},
                    }
                ],
            }
        if action == "execute":
            return {"status": "success", "trace": [{"tool": "write_file", "ok": True}]}
        if action == "abort":
            return {"status": "aborted", "discarded_snapshots": 0}
        return {"error": "unknown_action"}


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

    def test_instructions_param_flows_into_config(self):
        provider = FakeProvider(["ok"])
        agent = Agent(provider=provider, instructions=AGENT_MODES["precision"])
        agent.run("hi")
        self.assertEqual(len(provider.calls), 1)
        config = provider.calls[0][1]
        self.assertIn("write_file", config)
        self.assertNotIn("planner run", config)

    def test_instructions_can_switch_at_runtime(self):
        provider = FakeProvider(["ok", "ok"])
        agent = Agent(provider=provider)
        agent.run("hi")
        agent.instructions = AGENT_MODES["precision"]
        agent.run("hi")
        configs = [call[1] for call in provider.calls]
        self.assertIn("write_file", configs[0])
        self.assertIn("write_file", configs[1])
        self.assertNotIn("orchestrator run", configs[0])
        self.assertNotIn("planner run", configs[1])


class TestConfirmationFlow(unittest.TestCase):
    def _agent_with_orchestrator(self, provider):
        registry = build_default_registry()
        orch = FakeOrchestrator()
        registry.register("orchestrator", orch)
        agent = Agent(provider=provider, registry=registry)
        return agent, orch

    def _pending(self, provider):
        agent, orch = self._agent_with_orchestrator(provider)
        agent.run("crie out.txt")
        return agent, orch

    def test_awaiting_confirmation_returns_plan_deterministically(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "crie out.txt"}}\n```',
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        result = agent.run("crie out.txt")
        self.assertIn("write out.txt", result)
        self.assertIn("Aguardando sua aprovação", result)
        self.assertEqual(orch.calls[0][0], "run")
        self.assertEqual(len(orch.calls), 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNotNone(agent._pending_call)
        self.assertIn("awaiting", agent._pending_call["preview"]["status"])
        history = agent.memory.get_history()
        self.assertIn("Aguardando sua aprovação", history[-1]["content"])

    def test_approval_resumes_without_model_guess(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "crie out.txt"}}\n```',
                "Pronto, executei.",
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        agent.run("crie out.txt")
        result = agent.run("sim")
        self.assertEqual(result, "Pronto, executei.")
        self.assertEqual(orch.calls[0], ("run", {"request": "crie out.txt"}))
        self.assertEqual(orch.calls[1], ("execute", {"confirm": True}))
        self.assertEqual(len(orch.calls), 2)
        self.assertIsNone(agent._pending_call)

    def test_approval_continues_with_more_tool_calls(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "x"}}\n```',
                '```json\n{"tool": "orchestrator", "action": "execute", "params": {"confirm": true}}\n```',
                "Terminei.",
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        agent.run("x")
        result = agent.run("pode ir")
        self.assertEqual(result, "Terminei.")
        # deterministic resume first, then the model keeps working
        self.assertEqual(orch.calls[1], ("execute", {"confirm": True}))
        self.assertEqual(orch.calls[2], ("execute", {"confirm": True}))
        self.assertEqual(len(orch.calls), 3)
        self.assertIsNone(agent._pending_call)

    def test_abort_cancels_pending(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "x"}}\n```',
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        agent.run("x")
        result = agent.run("não, cancela")
        self.assertIn("cancelled", result)
        self.assertEqual(orch.calls[1], ("abort", {}))
        self.assertIsNone(agent._pending_call)
        self.assertEqual(len(provider.calls), 1)

    def test_conditional_approval_goes_to_model(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "crie out.txt"}}\n```',
                '```json\n{"tool": "orchestrator", "action": "execute", "params": {"confirm": true}}\n```',
                "Executei com as mudanças.",
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        agent.run("crie out.txt")
        result = agent.run("sim, mas troque o nome")
        self.assertEqual(result, "Executei com as mudanças.")
        guidance = provider.calls[1][0]
        self.assertIn("awaiting confirmation", guidance)
        self.assertIn("sim, mas troque o nome", guidance)
        self.assertEqual(orch.calls[1], ("execute", {"confirm": True}))

    def test_awaiting_on_last_iteration_still_returns_plan(self):
        provider = FakeProvider(
            [
                '```json\n{"tool": "orchestrator", "action": "run", "params": {"request": "x"}}\n```',
            ]
        )
        agent, orch = self._agent_with_orchestrator(provider)
        agent.max_iterations = 1
        result = agent.run("x")
        self.assertIn("write out.txt", result)
        self.assertNotIn("maximum", result)
        self.assertEqual(len(orch.calls), 1)
        self.assertIsNotNone(agent._pending_call)

    def test_tool_hint_is_forwarded_to_next_prompt(self):
        tool = type(
            "HintTool",
            (),
            {
                "name": "hinty",
                "get_manual": lambda self: "hinty manual\nActions:\nx",
                "dispatch": lambda self, action, **p: {
                    "status": "success",
                    "hint": "Quick lookup only. Continue with discovery run.",
                },
            },
        )()
        provider = FakeProvider(
            [
                '```json\n{"tool": "hinty", "action": "x", "params": {}}\n```',
                "final",
            ]
        )
        registry = build_default_registry()
        agent = Agent(provider=provider, registry=registry, extra_tools=[tool])
        result = agent.run("investigue")
        self.assertEqual(result, "final")
        self.assertIn(
            "Quick lookup only. Continue with discovery run.", provider.calls[1][0]
        )


class FlakyProvider:
    """Raises transient errors before finally serving a canned response."""

    def __init__(self, failures, response):
        self.failures = list(failures)
        self.response = response
        self.calls = 0

    def infer(self, user_prompt, config, **settings):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.response


class TestProviderTransientRetry(unittest.TestCase):
    def test_transient_error_is_retried_until_success(self):
        from src.providers.base import TransientProviderError

        provider = FlakyProvider(
            [TransientProviderError(429, "rate limited", retry_after=0)],
            "final answer",
        )
        agent = Agent(provider=provider, registry=build_default_registry())
        with mock.patch("src.agent.time.sleep") as sleep:
            result = agent.run("pergunta")
        self.assertEqual(result, "final answer")
        self.assertEqual(provider.calls, 2)
        sleep.assert_called()

    def test_non_transient_provider_error_propagates(self):
        from src.providers.base import ProviderError

        provider = FlakyProvider([ProviderError("HTTP 401: nope")], "")
        agent = Agent(provider=provider, registry=build_default_registry())
        with mock.patch("src.agent.time.sleep") as sleep:
            with self.assertRaises(ProviderError):
                agent.run("pergunta")
        sleep.assert_not_called()

    def test_exhausted_transient_retries_raise(self):
        from src.providers.base import TransientProviderError

        provider = FlakyProvider([TransientProviderError(500)] * 10, "")
        agent = Agent(provider=provider, registry=build_default_registry())
        with mock.patch("src.agent.time.sleep"):
            with self.assertRaises(TransientProviderError):
                agent.run("pergunta")
        self.assertEqual(provider.calls, agent._MAX_PROVIDER_RETRIES + 1)


if __name__ == "__main__":
    unittest.main()
