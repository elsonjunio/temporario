import unittest

from src.memory import Memory
from src.utils import (
    AGENT_MODES,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MODE,
    build_config_prompt,
    build_environment_info,
    classify_followup,
    parse_tool_call,
)


class TestMemory(unittest.TestCase):
    def test_add_and_history(self):
        memory = Memory()
        memory.add_user("hello")
        memory.add_assistant("hi")
        history = memory.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_add_tool(self):
        memory = Memory()
        memory.add_tool("list_dir", "list", {"path": "."}, {"total": 1})
        self.assertEqual(len(memory), 1)
        context = memory.get_context()
        self.assertIn("tool=list_dir", context)

    def test_context_limits_entries(self):
        memory = Memory()
        for i in range(10):
            memory.add_user(str(i))
        self.assertEqual(memory.get_context(max_entries=3).count("[user]"), 3)

    def test_clear(self):
        memory = Memory()
        memory.add_user("x")
        memory.clear()
        self.assertEqual(len(memory), 0)


class TestBuildConfigPrompt(unittest.TestCase):
    def test_manual_included(self):
        config = build_config_prompt(["tool A manual"])
        self.assertIn("tool A manual", config)
        self.assertIn("## Available tools", config)

    def test_memory_included(self):
        memory = Memory()
        memory.add_user("who are you")
        config = build_config_prompt("manual", memory=memory)
        self.assertIn("who are you", config)

    def test_default_instructions(self):
        config = build_config_prompt("manual")
        self.assertIn('{"tool"', config)

    def test_agent_modes_exist(self):
        self.assertEqual(set(AGENT_MODES), {"fast", "balanced", "precision"})
        self.assertEqual(DEFAULT_MODE, "fast")
        self.assertEqual(DEFAULT_INSTRUCTIONS, AGENT_MODES["fast"])

    def test_base_instructions_forbid_prose_tool_calls(self):
        text = AGENT_MODES["fast"]
        self.assertIn("Never describe a tool call in plain prose", text)
        self.assertIn("emit ONLY the JSON block", text)

    def _assert_base_only(self, text):
        self.assertIn("QUESTION / ANALYSIS", text)
        self.assertIn("an evaluation never changes files", text)
        self.assertIn("write_file", text)
        self.assertIn("navigation", text)
        # Must not instruct the model to INVOKE any disabled subagent.
        for invoke in (
            "orchestrator run",
            "discovery run",
            "planner run",
            "executor run",
            "REPLAN_REQUIRED",
        ):
            self.assertNotIn(invoke, text)

    def _assert_orch_routing(self, text):
        self.assertIn("QUESTION / ANALYSIS", text)
        self.assertIn("an evaluation never changes files", text)
        self.assertIn("navigation", text)
        # Changes are delegated to the orchestrator.
        self.assertIn("'orchestrator'", text)
        self.assertIn("confirm=true", text)
        # Other subagents stay disabled; never instruct the model to
        # INVOKE them directly.
        for invoke in (
            "discovery run",
            "planner run",
            "executor run",
            "REPLAN_REQUIRED",
        ):
            self.assertNotIn(invoke, text)

    def test_fast_mode_instructions(self):
        self._assert_orch_routing(AGENT_MODES["fast"])

    def test_fast_mode_routes_analysis_to_base_tools(self):
        text = AGENT_MODES["fast"]
        self.assertIn("QUESTION / ANALYSIS", text)
        self.assertIn("an evaluation never changes files", text)
        self.assertNotIn("write_file/patch_file/", text)

    def test_balanced_mode_instructions(self):
        self._assert_orch_routing(AGENT_MODES["balanced"])

    def test_precision_mode_instructions(self):
        text = AGENT_MODES["precision"]
        self._assert_base_only(text)
        # precision has no orchestrator at all
        self.assertNotIn("orchestrator' tool", text)

    def test_build_config_prompt_with_custom_instructions(self):
        config = build_config_prompt("manual", instructions=AGENT_MODES["precision"])
        self.assertIn("write_file", config)
        self.assertNotIn("planner run", config)

    def test_environment_included(self):
        env = build_environment_info(workspace="/tmp/ws")
        config = build_config_prompt("manual", environment=env)
        self.assertIn("## Environment", config)
        self.assertIn("OS:", config)
        self.assertIn("Workspace: /tmp/ws", config)

    def test_environment_optional(self):
        config = build_config_prompt("manual")
        self.assertNotIn("## Environment", config)

    def test_environment_info_no_workspace(self):
        env = build_environment_info()
        self.assertIn("OS:", env)
        self.assertNotIn("Workspace:", env)


class TestParseToolCall(unittest.TestCase):
    def test_fenced_block(self):
        text = 'Let me look.\n```json\n{"tool": "list_dir", "action": "list", "params": {"path": "."}}\n```'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["action"], "list")
        self.assertEqual(call["params"]["path"], ".")

    def test_plain_answer_returns_none(self):
        self.assertIsNone(
            parse_tool_call("This is the final answer, no tool call here.")
        )

    def test_missing_tool_key_returns_none(self):
        self.assertIsNone(parse_tool_call('```json\n{"foo": "bar"}\n```'))

    def test_unicode_text(self):
        text = 'preciso listar\n```json\n{"tool": "list_dir", "action": "list", "params": {}}\n```'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "list_dir")

    def test_invoke_xml_with_named_params(self):
        text = (
            '<invoke name="read_file">\n'
            '<parameter name="file_path">src/agent.py</parameter>\n'
            "</invoke>"
        )
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "read_file")
        self.assertEqual(call["action"], "read")
        self.assertEqual(call["params"]["file_path"], "src/agent.py")

    def test_invoke_xml_infers_action(self):
        text = '<invoke name="list_dir"><parameter name="path">.</parameter></invoke>'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["action"], "list")

    def test_invoke_xml_action_in_params(self):
        text = (
            '<invoke name="run_command">'
            '<parameter name="action">run</parameter>'
            '<parameter name="command">ls</parameter>'
            "</invoke>"
        )
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "run_command")
        self.assertEqual(call["action"], "run")
        self.assertNotIn("action", call["params"])

    def test_invoke_xml_bare_parameter_pairs(self):
        text = (
            '<invoke name="list_dir">\n'
            "<parameter>path</parameter>\n"
            "<parameter>.</parameter>\n"
            "</invoke>"
        )
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "list_dir")
        self.assertEqual(call["action"], "list")
        self.assertEqual(call["params"]["path"], ".")

    def test_invoke_xml_unknown_tool_defaults_to_run(self):
        text = '<invoke name="custom_thing"><parameter name="x">1</parameter></invoke>'
        call = parse_tool_call(text)
        self.assertEqual(call["tool"], "custom_thing")
        self.assertEqual(call["action"], "run")

    def test_plain_answer_without_invoke_returns_none(self):
        text = "<invoke>like a normal mention, not a tool call</invoke>"
        self.assertIsNone(parse_tool_call(text))


class TestClassifyFollowup(unittest.TestCase):
    def test_pure_approvals(self):
        for message in (
            "sim",
            "sim!",
            "sim por favor",
            "pode",
            "pode ir",
            "pode seguir",
            "ok",
            "ok, pode",
            "continue",
            "prossiga",
            "vai",
            "yes",
            "go ahead",
        ):
            self.assertEqual(
                classify_followup(message),
                "approve",
                f"expected approve for {message!r}",
            )

    def test_explicit_cancellations(self):
        for message in ("não", "nao", "no", "cancela", "aborta", "para", "stop"):
            self.assertEqual(
                classify_followup(message), "abort", f"expected abort for {message!r}"
            )

    def test_ambiguous_goes_to_other(self):
        for message in (
            "sim, mas troque o nome",
            "continue mas devagar",
            "pode ir, porém mude o arquivo",
            "o que você vai fazer?",
            "não sei",
            "",
            "   ",
            "conte mais sobre o plano",
            "não continua",
        ):
            self.assertEqual(
                classify_followup(message), "other", f"expected other for {message!r}"
            )

    def test_abort_takes_precedence(self):
        self.assertEqual(classify_followup("não"), "abort")
        self.assertEqual(classify_followup("no"), "abort")
        self.assertEqual(classify_followup("não, cancela"), "abort")
        self.assertEqual(classify_followup("para tudo"), "abort")


if __name__ == "__main__":
    unittest.main()
