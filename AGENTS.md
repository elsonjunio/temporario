# Agent Instructions

## Architecture
This repository implements an LLM agent that drives filesystem tools, plus a registry for MCP providers.

- **Tool Registry**: `src/tools/registry.py` defines `ToolRegistry` (register/unregister/dispatch/get_manual/list_tools) and `build_default_registry()`. The base tools are `read_file`, `list_dir`, `search_files`, `grep_files`, `write_file`, `patch_file`, `delete_file`, `move_file`, `run_command`. `src/tools/facade.py` is a backward-compatible facade over the `default_registry` (module-level `dispatch`/`get_manual`/`list_tools`, plus `register_tool`/`unregister_tool`). Each tool module exposes `get_manual()` and `dispatch(action, **params)`. Shared filesystem helpers live in `src/tools/_fs.py`.
- **Agentic Orchestrator**: `src/orchestrator/` is a *special tool* named `orchestrator`, built via `create_orchestrator_tool(registry, provider, memory, ...)` and registered into a `ToolRegistry` (see `src/main.py`). It drives discovery → planning → step-by-step execution → validation using only the tools in the registry. Actions: `run`, `discover`, `assess`, `plan`, `execute`, `pending`, `validate`, `undo`, `abort`. Mutations are snapshotted (`undo.py`) and rolled back automatically on failure; poor discovery aborts before planning. Because it is just another registry entry, removing it keeps the agent working with the base tools — low coupling.
  - **Replace vs alter**: `src/orchestrator/impact.py` (`ImpactAssessor`) profiles each target deterministically — does it exist, is it a registered tool module, is the module contract (`MANUAL`, `SPEC`, `get_manual`, `dispatch`) intact — and classifies the change as `new` / `modify`. Existing files are planned via `patch_file`; `write_file` is reserved for new files or explicit rewrites (`"rewrite": true`). A plan that would `write_file` over an existing registered tool module without `rewrite: true` is rejected with guidance (re-plan). Rewrites are deep-validated at execution: the module is re-imported and the contract re-checked, otherwise everything rolls back.
  - **Tests in the plan**: `ImpactAssessor` detects each target's language (python, typescript, javascript, java, kotlin, c, cpp, go, rust, ruby, php, csharp, swift, …), maps existing unit tests by the language's conventions and detects the project's test runner (package.json→jest, pom.xml/gradlew→mvn/gradle, go.mod→go test, Cargo.toml→cargo test, …). When a changed module is covered by tests, the planner adds a final `run_command` step running that command with the language's success marker. When a framework is configured but nothing covers the target, the planner adds steps to create a suggested test file and run it. When no test setup exists, the planner surfaces a recommendation and does NOT auto-create a harness — it implements a test only when the user asks for one.
- **Agent Loop**: `src/agent.py` implements `Agent.run(user_prompt)` which iterates: build config prompt → `provider.infer` → parse tool call → `registry.dispatch` → record result, until a plain answer or `max_iterations`. The agent accepts a custom `registry` and optional `extra_tools`. **Confirmation flow**: when a dispatched result is `awaiting_confirmation`, the agent stores a `_pending_call` and prompts the model to relay the plan. On the next user turn it classifies the reply via `classify_followup` (`src/utils.py`): pure approvals ("sim", "pode ir", "continue") resume deterministically via `execute(confirm=True)` without a model round-trip; explicit cancellations call `abort`; ambiguous/conditional replies ("sim, mas muda X") go to the model with the pending plan as context. The orchestrator exposes `pending` to read the pending plan and `execute(confirm=True)` to resume it (no steps needed).
- **Provider Client**: `src/providers/opencode.py` implements `OpenCodeProvider`, an OpenAI-compatible client for OpenCode Zen (default model `big-pickle`, auth via `OPENCODE_API_KEY`). `infer(user_prompt, config)` sends a two-part prompt.
- **Config Prompt**: `src/utils.py::build_config_prompt()` assembles the configuration/system prompt (runtime environment via `build_environment_info()` — OS, python, workspace — plus tool list from the manuals, instructions and conversation history). `parse_tool_call()` extracts the tool-call JSON block from the model output. `extract_json_object()` is a general JSON extractor.
- **Memory**: `src/memory.py` defines the `Memory` class that stores conversation history and tool results. An optional `ContextCompressor` (from `src/context.py`) observes every add and summarizes old entries once the estimated token count reaches a threshold. Future work: on-demand tool context.
- **MCP Registry**: The `Registry` class in `src/registry.py` scans a directory for subdirectories containing a `run.sh` (Linux) or `run.bat` (Windows) file.

## Adding a New Tool
1. Create `src/tools/<name>.py` defining the handler(s).
2. Define `MANUAL` (params and purpose, for the LLM) and a module-level `ToolSpec` with the action→handler map.
3. Export `get_manual()` and `dispatch(action, **params)`.
4. Register the module in `src/tools/registry.py::build_default_registry()` (or at runtime via `register_tool`).

## Using / Removing the Orchestrator
- Register: `create_orchestrator_tool(registry, provider, memory)` then `registry.register(tool.name, tool)`.
- Remove: `registry.unregister("orchestrator")` — the agent continues with the base tools unchanged.

## Adding a New MCP Provider
1. Create a new directory within the providers root.
2. Add a `run.sh` (for Linux) or `run.bat` (for Windows) file to that directory.
3. Ensure the script is executable and starts the desired service.

## Context Compression
`ContextCompressor` (in `src/context.py`) monitors the estimated token count of the memory and, once it reaches `CONTEXT_TOKEN_THRESHOLD`, summarizes the oldest entries (outside a recency buffer) via the provider and replaces them with a single `summary` entry. If the provider fails, it degrades to relevance-based truncation (n-gram embeddings). Wire it with `Memory(compressor=ContextCompressor(provider=provider))`; leaving the compressor off preserves the old behavior. Env vars:

- `CONTEXT_TOKEN_THRESHOLD` (default `6000`; `0` disables compression)
- `CONTEXT_TOKEN_FACTOR` (default `2.5`) — fallback estimator: words × factor
- `CONTEXT_KEEP_RECENT` (default `4`) — newest entries never summarized
- `CONTEXT_MAX_SUMMARY_CHARS` (default `2000`) — summary length cap

## Development & Testing
- **Dependencies**: Requires `mcp` (see `requirements.txt`). Runtime tools use only the Python stdlib.
- **Environment**: Uses a Python virtual environment (`.venv`).
- **Tests**: `unittest` — run with `.venv/bin/python -m unittest discover -s tests`.
- **Type checking**: `.venv/bin/python -m mypy src tests`.
- **Formatting**: `.venv/bin/python -m black src tests`.
- **Demo**: `.venv/bin/python src/main.py` (needs `OPENCODE_API_KEY` for the live agent demo).
